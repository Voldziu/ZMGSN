"""Report generation: bibliographies and end-to-end literature reviews."""

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
from tool_eval.utils.text import append_text, write_text

logger = get_logger(__name__)

# Fixed so every tool level and the evaluation checks agree on where the review lives.
REVIEW_PATH = "reviews/review.md"


class ReviewResult(BaseModel):
    """Outcome of a literature review run."""

    model_config = ConfigDict(frozen=True)

    review_path: str
    papers_included: int
    papers_failed: list[str]


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
        Papers, newest first.
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


def append_to_review(markdown: str, *, settings: Settings) -> str:
    """Append Markdown to the review file, creating it if needed.

    Args:
        markdown: Text to append.
        settings: Application settings.

    Returns:
        Workspace-relative path of the review file.
    """
    dest = resolve_in_workspace(REVIEW_PATH, workspace=settings.workspace_dir)
    append_text(dest, markdown.rstrip() + "\n\n", max_bytes=settings.max_file_bytes)
    return to_ref(dest, workspace=settings.workspace_dir)


def append_bibliography(arxiv_ids: Sequence[str], *, settings: Settings) -> str:
    """Append a `## Bibliography` section with sorted APA references to the review file.

    Args:
        arxiv_ids: arXiv identifiers.
        settings: Application settings.

    Returns:
        Workspace-relative path of the review file.
    """
    papers = fetch_arxiv_papers(arxiv_ids, timeout=settings.http_timeout_seconds)
    entries = sorted(_citation(paper) for paper in papers)
    section = "## Bibliography\n\n" + "\n".join(f"- {entry}" for entry in entries)
    return append_to_review(section, settings=settings)


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
    """Search arXiv, summarize the top papers and write a Markdown review with a bibliography.

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
    papers = search_arxiv(
        query,
        max_results=settings.review_paper_count,
        date_from=settings.arxiv_date_from,
        date_to=settings.arxiv_date_to,
        timeout=settings.http_timeout_seconds,
    )
    if not papers:
        msg = "No papers found for this query in the configured date window."
        raise InvalidInputError(msg)

    sections: list[str] = []
    citations: list[str] = []
    failed: list[str] = []
    for paper in papers:
        try:
            result = summarize_paper(paper.arxiv_id, settings=settings, llm=llm)
        except ToolEvalError:
            logger.warning("Could not summarize %s", paper.arxiv_id, exc_info=True)
            failed.append(paper.arxiv_id)
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
    return ReviewResult(
        review_path=to_ref(dest, workspace=settings.workspace_dir),
        papers_included=len(sections),
        papers_failed=failed,
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
