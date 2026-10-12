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
    """Cut `text` to at most `max_chars` characters.

    Args:
        text: Text to shorten.
        max_chars: Maximum number of characters to keep.

    Returns:
        `text` unchanged if short enough, otherwise its first `max_chars` characters.
    """
    return text if len(text) <= max_chars else text[:max_chars]


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
    if not 0 <= overlap < chunk_size:
        msg = "overlap must satisfy 0 <= overlap < chunk_size."
        raise InvalidInputError(msg)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + chunk_size // 2, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _check_size(content: str, *, existing: int, max_bytes: int, name: str) -> None:
    if existing + len(content.encode("utf-8")) > max_bytes:
        msg = f"File would exceed the size limit: {name}"
        raise WorkspaceError(msg)
