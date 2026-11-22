"""Workspace-based paper operations. Level 1 tools call them one by one; levels 2-3 chain them."""

import shutil
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from tool_eval.config import Settings
from tool_eval.exceptions import InvalidInputError, PaperNotFoundError, ToolEvalError
from tool_eval.prompts import load_prompt
from tool_eval.services import llm_ops
from tool_eval.services.arxiv import fetch_arxiv_papers, validate_arxiv_id
from tool_eval.services.network import count_network_request
from tool_eval.services.pdf import download_arxiv_pdf, extract_pdf_text
from tool_eval.services.reading import record_text_read, write_chunk_index
from tool_eval.utils.logger import get_logger
from tool_eval.utils.paths import resolve_in_workspace, to_ref
from tool_eval.utils.text import read_text, split_spans, write_text

logger = get_logger(__name__)


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
        f"papers/{paper_file_stem(cleaned)}.pdf",
        workspace=settings.workspace_dir,
    )
    cache = settings.pdf_cache_dir
    source = dest if cache is None else cache / dest.name
    missing = None if cache is None else cache / f"{dest.stem}.missing"
    if missing is not None and missing.is_file():
        raise PaperNotFoundError(missing.read_text(encoding="utf-8"))
    if not source.is_file():
        try:
            _download(cleaned, dest=source, settings=settings)
        except PaperNotFoundError as exc:
            if missing is not None:
                missing.parent.mkdir(parents=True, exist_ok=True)
                missing.write_text(str(exc), encoding="utf-8")
            raise
    if source != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    return to_ref(dest, workspace=settings.workspace_dir)


def _download(arxiv_id: str, *, dest: Path, settings: Settings) -> None:
    """Download a PDF; on 404 try once more at the paper's latest version."""
    try:
        count_network_request("pdf_downloads")
        download_arxiv_pdf(
            arxiv_id,
            dest=dest,
            timeout=settings.http_timeout_seconds,
            max_bytes=settings.max_pdf_bytes,
        )
    except PaperNotFoundError as not_found:
        try:
            found = fetch_arxiv_papers([arxiv_id], timeout=settings.http_timeout_seconds)
        except ToolEvalError:
            raise not_found from None
        version = found[0].version if found else None
        if version is None:
            raise
        count_network_request("pdf_downloads")
        download_arxiv_pdf(
            arxiv_id,
            dest=dest,
            timeout=settings.http_timeout_seconds,
            max_bytes=settings.max_pdf_bytes,
            version=version,
        )


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
    text = read_text(source, max_bytes=settings.max_file_bytes)
    spans = split_spans(text, chunk_size=settings.chunk_size)
    refs: list[str] = []
    index: dict[str, tuple[int, int]] = {}
    for number, (start, end) in enumerate(spans[: settings.pipeline_max_chunks], start=1):
        dest = resolve_in_workspace(f"chunks/{source.stem}/{number:04d}.txt", workspace=workspace)
        write_text(dest, text[start:end], max_bytes=settings.max_file_bytes)
        refs.append(to_ref(dest, workspace=workspace))
        index[dest.name] = (start, end)
    write_chunk_index(resolve_in_workspace(f"chunks/{source.stem}", workspace=workspace), index)
    return refs, len(spans)


def summarize_chunk_file(chunk_ref: str, *, settings: Settings, llm: BaseChatModel) -> str:
    """Summarize one chunk (a text file of at most `chunk_size` characters) with the LLM.

    Args:
        chunk_ref: Workspace-relative path of the text file.
        settings: Application settings.
        llm: Chat model.

    Returns:
        The summary.

    Raises:
        InvalidInputError: If the file is longer than one chunk; nothing is sent to the model.
    """
    path = resolve_in_workspace(chunk_ref, workspace=settings.workspace_dir)
    text = read_text(path, max_bytes=settings.max_file_bytes)
    if len(text) > settings.chunk_size:
        msg = (
            f"This file has {len(text)} characters; a chunk holds at most {settings.chunk_size}. "
            "Split it with split_text_into_chunks and summarize the chunks."
        )
        raise InvalidInputError(msg)
    record_text_read(path, workspace=settings.workspace_dir, kept=len(text))
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
    kept = min(len(text), settings.text_max_chars)
    record_text_read(path, workspace=settings.workspace_dir, kept=kept)
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
    record_text_read(source, workspace=workspace, kept=min(len(text), settings.text_max_chars))
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
    """Download a paper, summarize its leading chunks and combine them into one summary.

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
    prompt = load_prompt("synthesize_summary", settings.prompt_version)
    summary = llm_ops.synthesize(parts, llm=llm, prompt=prompt)
    dest = resolve_in_workspace(
        f"summaries/{paper_file_stem(cleaned)}.md",
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


def paper_file_stem(arxiv_id: str) -> str:
    """File-name stem the tools use for a paper (`papers/<stem>.pdf`, `texts/<stem>.txt`).

    Args:
        arxiv_id: arXiv identifier.

    Returns:
        The identifier with `/` (old-style ids) replaced by `_`.
    """
    return arxiv_id.replace("/", "_")
