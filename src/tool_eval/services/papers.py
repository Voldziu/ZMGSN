"""Workspace-based paper operations. Level 1 tools call them one by one; levels 2-3 chain them."""

import logging
import shutil

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from tool_eval.config import Settings
from tool_eval.exceptions import InvalidInputError
from tool_eval.prompts import load_prompt
from tool_eval.services import llm_ops
from tool_eval.services.arxiv import validate_arxiv_id
from tool_eval.services.pdf import download_arxiv_pdf, extract_pdf_text
from tool_eval.utils.paths import resolve_in_workspace, to_ref
from tool_eval.utils.text import read_text, split_text, write_text

logger = logging.getLogger(__name__)


class PaperSummary(BaseModel):
    """Result of summarizing one paper."""

    model_config = ConfigDict(frozen=True)

    arxiv_id: str
    summary_path: str
    summary: str
    chunks_summarized: int
    chunks_total: int


def download_paper_pdf(arxiv_id: str, *, settings: Settings) -> str:
    """Download a paper's PDF into `papers/` in the workspace.

    Args:
        arxiv_id: arXiv identifier.
        settings: Application settings.

    Returns:
        Workspace-relative path of the PDF.
    """
    cleaned = validate_arxiv_id(arxiv_id)
    dest = resolve_in_workspace(
        f"papers/{_safe_stem(cleaned)}.pdf",
        workspace=settings.workspace_dir,
    )
    cache = settings.pdf_cache_dir
    source = dest if cache is None else cache / dest.name
    if not source.is_file():
        download_arxiv_pdf(
            cleaned,
            dest=source,
            timeout=settings.http_timeout_seconds,
            max_bytes=settings.max_file_bytes,
        )
    if source != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    return to_ref(dest, workspace=settings.workspace_dir)


def extract_paper_text(pdf_ref: str, *, settings: Settings) -> str:
    """Extract a PDF's text into `texts/` in the workspace.

    Args:
        pdf_ref: Workspace-relative path of the PDF.
        settings: Application settings.

    Returns:
        Workspace-relative path of the text file.

    Raises:
        InvalidInputError: If the PDF has no extractable text.
    """
    workspace = settings.workspace_dir
    pdf_path = resolve_in_workspace(pdf_ref, workspace=workspace)
    text = extract_pdf_text(pdf_path)
    if not text:
        msg = f"No extractable text in {pdf_path.name}; it may be a scanned document."
        raise InvalidInputError(msg)
    dest = resolve_in_workspace(f"texts/{pdf_path.stem}.txt", workspace=workspace)
    write_text(dest, text, max_bytes=settings.max_file_bytes)
    return to_ref(dest, workspace=workspace)


def split_paper_text(text_ref: str, *, settings: Settings) -> list[str]:
    """Split a text file into chunk files under `chunks/<name>/`.

    Only the first `pipeline_max_chunks` chunks are written, so every tool level works from
    the same material.

    Args:
        text_ref: Workspace-relative path of the text file.
        settings: Application settings.

    Returns:
        Workspace-relative paths of the chunk files, in order.
    """
    return _split_paper_text(text_ref, settings=settings)[0]


def _split_paper_text(text_ref: str, *, settings: Settings) -> tuple[list[str], int]:
    workspace = settings.workspace_dir
    source = resolve_in_workspace(text_ref, workspace=workspace)
    chunks = split_text(
        read_text(source, max_bytes=settings.max_file_bytes),
        chunk_size=settings.chunk_size,
    )
    refs: list[str] = []
    for number, chunk in enumerate(chunks[: settings.pipeline_max_chunks], start=1):
        dest = resolve_in_workspace(f"chunks/{source.stem}/{number:04d}.txt", workspace=workspace)
        write_text(dest, chunk, max_bytes=settings.max_file_bytes)
        refs.append(to_ref(dest, workspace=workspace))
    return refs, len(chunks)


def summarize_chunk_file(chunk_ref: str, *, settings: Settings, llm: BaseChatModel) -> str:
    """Summarize one text file with the LLM.

    Args:
        chunk_ref: Workspace-relative path of the text file.
        settings: Application settings.
        llm: Chat model.

    Returns:
        The summary.
    """
    path = resolve_in_workspace(chunk_ref, workspace=settings.workspace_dir)
    text = read_text(path, max_bytes=settings.max_file_bytes)
    prompt = load_prompt("summarize_chunk", settings.prompt_version)
    return llm_ops.summarize(text, llm=llm, prompt=prompt, max_chars=settings.text_max_chars)


def keywords_from_file(
    text_ref: str,
    *,
    max_keywords: int,
    settings: Settings,
    llm: BaseChatModel,
) -> list[str]:
    """Extract keywords from a text file with the LLM.

    Args:
        text_ref: Workspace-relative path of the text file.
        max_keywords: Maximum number of keywords.
        settings: Application settings.
        llm: Chat model.

    Returns:
        The keywords.
    """
    path = resolve_in_workspace(text_ref, workspace=settings.workspace_dir)
    text = read_text(path, max_bytes=settings.max_file_bytes)
    prompt = load_prompt("extract_keywords", settings.prompt_version)
    return llm_ops.extract_keywords(
        text,
        llm=llm,
        prompt=prompt,
        max_keywords=max_keywords,
        max_chars=settings.text_max_chars,
    )


def translate_file(
    text_ref: str,
    *,
    target_language: str,
    settings: Settings,
    llm: BaseChatModel,
) -> str:
    """Translate a text file and store the result under `translations/`.

    Args:
        text_ref: Workspace-relative path of the text file.
        target_language: Language name, for example `Polish`.
        settings: Application settings.
        llm: Chat model.

    Returns:
        Workspace-relative path of the translated file.

    Raises:
        InvalidInputError: If the language name is empty or not alphabetic.
    """
    language = target_language.strip()
    if not language.replace(" ", "").isalpha():
        msg = "target_language must be a language name such as 'Polish'."
        raise InvalidInputError(msg)
    workspace = settings.workspace_dir
    source = resolve_in_workspace(text_ref, workspace=workspace)
    text = read_text(source, max_bytes=settings.max_file_bytes)
    prompt = load_prompt("translate_text", settings.prompt_version)
    translated = llm_ops.translate(
        text,
        llm=llm,
        prompt=prompt,
        target_language=language,
        max_chars=settings.text_max_chars,
    )
    dest = resolve_in_workspace(
        f"translations/{source.stem}.{language.lower().replace(' ', '-')}.txt",
        workspace=workspace,
    )
    write_text(dest, translated, max_bytes=settings.max_file_bytes)
    return to_ref(dest, workspace=workspace)


def summarize_paper(arxiv_id: str, *, settings: Settings, llm: BaseChatModel) -> PaperSummary:
    """Download a paper, extract and split its text, and summarize its leading chunks.

    Args:
        arxiv_id: arXiv identifier.
        settings: Application settings.
        llm: Chat model.

    Returns:
        The summary and where it was stored (`summaries/<id>.md`).
    """
    cleaned = validate_arxiv_id(arxiv_id)
    pdf_ref = download_paper_pdf(cleaned, settings=settings)
    chunk_refs, total = _split_paper_text(
        extract_paper_text(pdf_ref, settings=settings),
        settings=settings,
    )
    parts = [summarize_chunk_file(ref, settings=settings, llm=llm) for ref in chunk_refs]
    summary = "\n\n".join(parts)
    dest = resolve_in_workspace(
        f"summaries/{_safe_stem(cleaned)}.md",
        workspace=settings.workspace_dir,
    )
    write_text(dest, summary, max_bytes=settings.max_file_bytes)
    return PaperSummary(
        arxiv_id=cleaned,
        summary_path=to_ref(dest, workspace=settings.workspace_dir),
        summary=summary,
        chunks_summarized=len(chunk_refs),
        chunks_total=total,
    )


def _safe_stem(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")
