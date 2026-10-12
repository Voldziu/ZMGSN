"""Path helpers that keep every file operation inside the workspace sandbox."""

from pathlib import Path

from tool_eval.exceptions import WorkspaceError

ALLOWED_TEXT_SUFFIXES = frozenset({".md", ".txt"})


def resolve_in_workspace(ref: str, *, workspace: Path) -> Path:
    """Resolve a workspace-relative reference to an absolute path inside the workspace.

    Args:
        ref: Relative path supplied by the caller (for example by the LLM).
        workspace: Workspace root directory.

    Returns:
        The resolved absolute path, guaranteed to be inside `workspace`.

    Raises:
        WorkspaceError: If the path is empty, absolute, malformed or escapes the workspace.
    """
    if not ref.strip():
        msg = "Path must not be empty."
        raise WorkspaceError(msg)
    candidate = Path(ref)
    if candidate.is_absolute():
        msg = "Absolute paths are not allowed; use a path relative to the workspace."
        raise WorkspaceError(msg)
    root = workspace.resolve()
    try:
        resolved = (root / candidate).resolve()
    except (OSError, ValueError) as exc:
        msg = "Invalid path."
        raise WorkspaceError(msg) from exc
    if not resolved.is_relative_to(root):
        msg = "Path escapes the workspace."
        raise WorkspaceError(msg)
    return resolved


def to_ref(path: Path, *, workspace: Path) -> str:
    """Convert an absolute path inside the workspace to a POSIX-style relative reference.

    Args:
        path: Absolute path located inside `workspace`.
        workspace: Workspace root directory.

    Returns:
        The reference string handed to the LLM.
    """
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def require_text_suffix(path: Path) -> None:
    """Ensure the path has an allowed plain-text suffix.

    Args:
        path: Path to check.

    Raises:
        WorkspaceError: If the suffix is not one of `ALLOWED_TEXT_SUFFIXES`.
    """
    if path.suffix.lower() not in ALLOWED_TEXT_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_TEXT_SUFFIXES))
        msg = f"Only these file types are allowed: {allowed}."
        raise WorkspaceError(msg)
