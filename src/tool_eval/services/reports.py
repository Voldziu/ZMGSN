"""Report generation: bibliographies and end-to-end literature reviews."""

import re
import shutil
from collections.abc import Sequence
from datetime import date

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from tool_eval.config import Settings
from tool_eval.exceptions import ExternalServiceError, InvalidInputError, ToolEvalError
from tool_eval.services.arxiv import ArxivPaper, fetch_arxiv_papers, search_arxiv
from tool_eval.services.formatting import format_apa_citation
from tool_eval.services.papers import summarize_paper
from tool_eval.utils.logger import get_logger
from tool_eval.utils.paths import require_text_suffix, resolve_in_workspace, to_ref
from tool_eval.utils.text import read_text, write_text

logger = get_logger(__name__)

# Fixed so every tool level and the evaluation checks agree on where the review lives.
REVIEW_PATH = "reviews/review.md"
# Run-workspace folders the level-3 tool fills; cleared at the start of every call.
_PAPER_FOLDERS = ("papers", "texts", "chunks", "summaries")
# Level 3 searches this many candidates per requested paper.
CANDIDATE_FACTOR = 2
_BIBLIOGRAPHY_HEADING = re.compile(r"^##[ \t]+Bibliography[ \t]*$", re.MULTILINE | re.IGNORECASE)


class SkippedPaper(BaseModel):
    """A search candidate the review could not read, and why."""

    model_config = ConfigDict(frozen=True)

    arxiv_id: str
    reason: str


class ReviewResult(BaseModel):
    """Outcome of a literature review run."""

    model_config = ConfigDict(frozen=True)

    message: str
    complete: bool
    review_path: str
    papers_requested: int
    papers_included: int
    papers_skipped: list[SkippedPaper]


class CitedPaper(BaseModel):
    """A search hit with its ready-made APA reference."""

    model_config = ConfigDict(frozen=True)

    arxiv_id: str
    title: str
    published: date
    apa_citation: str


def search_cited_papers(query: str, *, max_results: int, settings: Settings) -> list[CitedPaper]:
    """Search arXiv inside the configured date window and format each hit as an APA reference.

    Args:
        query: Keywords describing the topic.
        max_results: Maximum number of papers.
        settings: Application settings.

    Returns:
        Papers, most relevant first.
    """
    papers = search_arxiv(
        query,
        max_results=max_results,
        date_from=settings.arxiv_date_from,
        date_to=settings.arxiv_date_to,
        timeout=settings.http_timeout_seconds,
    )
    return [
        CitedPaper(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            published=paper.published,
            apa_citation=_citation(paper),
        )
        for paper in papers
    ]


def write_review(markdown: str, *, settings: Settings) -> str:
    """Replace the review file with `markdown`, creating it if needed.

    Args:
        markdown: The complete review.
        settings: Application settings.

    Returns:
        Workspace-relative path of the review file.
    """
    dest = resolve_in_workspace(REVIEW_PATH, workspace=settings.workspace_dir)
    write_text(dest, markdown.rstrip() + "\n", max_bytes=settings.max_file_bytes)
    return to_ref(dest, workspace=settings.workspace_dir)


def append_bibliography(arxiv_ids: Sequence[str], *, settings: Settings) -> str:
    """Write the `## Bibliography` section (sorted APA references) at the end of the review.

    An existing bibliography section, from its heading to the end of the file, is replaced, so
    calling this again never produces a second section; the text before it is kept.

    Args:
        arxiv_ids: arXiv identifiers.
        settings: Application settings.

    Returns:
        Workspace-relative path of the review file.
    """
    papers = fetch_arxiv_papers(arxiv_ids, timeout=settings.http_timeout_seconds)
    entries = sorted(_citation(paper) for paper in papers)
    section = "## Bibliography\n\n" + "\n".join(f"- {entry}" for entry in entries)
    dest = resolve_in_workspace(REVIEW_PATH, workspace=settings.workspace_dir)
    body = ""
    if dest.is_file():
        existing = read_text(dest, max_bytes=settings.max_file_bytes)
        body = _BIBLIOGRAPHY_HEADING.split(existing, maxsplit=1)[0].rstrip()
    return write_review(f"{body}\n\n{section}" if body else section, settings=settings)


def write_bibliography(arxiv_ids: Sequence[str], *, output_ref: str, settings: Settings) -> str:
    """Write an alphabetically sorted APA bibliography for the given papers.

    Args:
        arxiv_ids: arXiv identifiers.
        output_ref: Workspace-relative `.md` or `.txt` path to write.
        settings: Application settings.

    Returns:
        Workspace-relative path of the written file.
    """
    dest = resolve_in_workspace(output_ref, workspace=settings.workspace_dir)
    require_text_suffix(dest)
    papers = fetch_arxiv_papers(arxiv_ids, timeout=settings.http_timeout_seconds)
    entries = sorted(_citation(paper) for paper in papers)
    content = "# Bibliography\n\n" + "\n\n".join(f"- {entry}" for entry in entries) + "\n"
    write_text(dest, content, max_bytes=settings.max_file_bytes)
    return to_ref(dest, workspace=settings.workspace_dir)


def conduct_review(query: str, *, settings: Settings, llm: BaseChatModel) -> ReviewResult:
    """Search arXiv, summarize the top readable papers and write a Markdown review.

    Twice as many candidates as needed are searched; candidates that cannot be read (oversized
    or missing PDFs) are skipped and replaced by the next ones, so an unread paper never enters
    the review.

    Args:
        query: Research topic.
        settings: Application settings.
        llm: Chat model.

    Returns:
        Where the review was written and how many papers made it in.

    Raises:
        InvalidInputError: If no papers match the query in the configured date window.
        ExternalServiceError: If none of the found papers could be summarized.
    """
    # A repeated call starts clean, so the workspace holds only the papers of the last call.
    for folder in _PAPER_FOLDERS:
        shutil.rmtree(
            resolve_in_workspace(folder, workspace=settings.workspace_dir), ignore_errors=True
        )
    # Spare candidates replace papers that cannot be read (oversized or missing PDFs).
    papers = search_arxiv(
        query,
        max_results=CANDIDATE_FACTOR * settings.review_paper_count,
        date_from=settings.arxiv_date_from,
        date_to=settings.arxiv_date_to,
        timeout=settings.http_timeout_seconds,
    )
    if not papers:
        msg = "No papers found for this query in the configured date window."
        raise InvalidInputError(msg)

    sections: list[str] = []
    citations: list[str] = []
    skipped: list[SkippedPaper] = []
    for paper in papers:
        if len(sections) == settings.review_paper_count:
            break
        try:
            result = summarize_paper(paper.arxiv_id, settings=settings, llm=llm)
        except ToolEvalError as exc:
            logger.warning("Could not summarize %s", paper.arxiv_id, exc_info=True)
            skipped.append(SkippedPaper(arxiv_id=paper.arxiv_id, reason=str(exc)))
            continue
        citation = _citation(paper)
        citations.append(citation)
        sections.append(f"## {len(sections) + 1}. {paper.title}\n\n{citation}\n\n{result.summary}")
    if not sections:
        msg = "None of the found papers could be summarized."
        raise ExternalServiceError(msg)

    window = _describe_window(settings)
    bibliography = "\n\n".join(f"- {entry}" for entry in sorted(citations))
    content = (
        f"# Literature review: {query.strip()}\n\nSource: arXiv. {window}\n\n"
        + "\n\n".join(sections)
        + f"\n\n## Bibliography\n\n{bibliography}\n"
    )
    dest = resolve_in_workspace(REVIEW_PATH, workspace=settings.workspace_dir)
    write_text(dest, content, max_bytes=settings.max_file_bytes)
    path = to_ref(dest, workspace=settings.workspace_dir)
    requested, included = settings.review_paper_count, len(sections)
    complete = included == requested
    message = (
        f"Review with {included} papers read in full saved to {path}. The task is complete."
        if complete
        else f"Only {included} of {requested} papers could be read; the review saved to {path} "
        "is incomplete."
    )
    return ReviewResult(
        message=message,
        complete=complete,
        review_path=path,
        papers_requested=settings.review_paper_count,
        papers_included=len(sections),
        papers_skipped=skipped,
    )


def _citation(paper: ArxivPaper) -> str:
    return format_apa_citation(
        paper.authors,
        year=paper.published.year,
        title=paper.title,
        arxiv_id=paper.arxiv_id,
    )


def _describe_window(settings: Settings) -> str:
    start, end = settings.arxiv_date_from, settings.arxiv_date_to
    if start is None and end is None:
        return "No date filter."
    return f"Submitted between {start or 'the beginning'} and {end or 'today'}."
