"""Small text helpers: bounded file I/O, truncation and chunking."""

from pathlib import Path

from tool_eval.exceptions import InvalidInputError, WorkspaceError


def read_text(path: Path, *, max_bytes: int) -> str:
    """Read a UTF-8 text file, refusing files larger than `max_bytes`.

    Args:
        path: File to read.
        max_bytes: Maximum accepted file size in bytes.

    Returns:
        The file contents.

    Raises:
        WorkspaceError: If the file is missing, too large or not valid UTF-8.
    """
    if not path.is_file():
        msg = f"File not found: {path.name}"
        raise WorkspaceError(msg)
    if path.stat().st_size > max_bytes:
        msg = f"File is too large: {path.name}"
        raise WorkspaceError(msg)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        msg = f"File is not valid UTF-8 text: {path.name}"
        raise WorkspaceError(msg) from exc


def write_text(path: Path, content: str, *, max_bytes: int) -> None:
    """Write UTF-8 text to a file, creating parent directories.

    Args:
        path: Destination file.
        content: Text to write.
        max_bytes: Maximum accepted size of the content in bytes.

    Raises:
        WorkspaceError: If the content exceeds `max_bytes`.
    """
    _check_size(content, existing=0, max_bytes=max_bytes, name=path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_text(path: Path, content: str, *, max_bytes: int) -> None:
    """Append UTF-8 text to a file, creating it and its parents if needed.

    Args:
        path: Destination file.
        content: Text to append.
        max_bytes: Maximum accepted size of the resulting file in bytes.

    Raises:
        WorkspaceError: If the resulting file would exceed `max_bytes`.
    """
    existing = path.stat().st_size if path.is_file() else 0
    _check_size(content, existing=existing, max_bytes=max_bytes, name=path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(content)


def truncate(text: str, *, max_chars: int) -> str:
    """Cut `text` to at most `max_chars` characters and say so.

    Args:
        text: Text to shorten.
        max_chars: Maximum number of characters of `text` to keep.

    Returns:
        `text` unchanged if short enough, otherwise its first `max_chars` characters followed
        by a marker giving the characters kept and the total (see `cut_marker`).
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + cut_marker(max_chars, len(text))


def cut_marker(kept: int, total: int) -> str:
    """Marker appended to text that was cut, so the reader knows part of it is missing.

    Args:
        kept: Characters kept.
        total: Characters of the full text.

    Returns:
        The marker, starting with a blank line.
    """
    return f"\n\n[cut: showing {kept} of {total} characters]"


def split_text(text: str, *, chunk_size: int, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, preferring to break at spaces.

    Args:
        text: Text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Characters shared between consecutive chunks.

    Returns:
        Non-empty chunks in document order.

    Raises:
        InvalidInputError: If `overlap` is negative or not smaller than `chunk_size`.
    """
    return [
        text[start:end] for start, end in split_spans(text, chunk_size=chunk_size, overlap=overlap)
    ]


def split_spans(text: str, *, chunk_size: int, overlap: int = 200) -> list[tuple[int, int]]:
    """Character spans `[start, end)` of the chunks `split_text` returns.

    Args:
        text: Text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Characters shared between consecutive chunks.

    Returns:
        Spans of the non-empty, whitespace-stripped chunks in document order.

    Raises:
        InvalidInputError: If `overlap` is negative or not smaller than `chunk_size`.
    """
    if not 0 <= overlap < chunk_size:
        msg = "overlap must satisfy 0 <= overlap < chunk_size."
        raise InvalidInputError(msg)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + chunk_size // 2, end)
            if space != -1:
                end = space
        piece = text[start:end]
        stripped = piece.strip()
        if stripped:
            first = start + (len(piece) - len(piece.lstrip()))
            spans.append((first, first + len(stripped)))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return spans


def _check_size(content: str, *, existing: int, max_bytes: int, name: str) -> None:
    if existing + len(content.encode("utf-8")) > max_bytes:
        msg = f"File would exceed the size limit: {name}"
        raise WorkspaceError(msg)
