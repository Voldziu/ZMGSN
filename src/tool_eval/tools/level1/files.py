"""File tools restricted to `.md` and `.txt` files inside the workspace."""

from langchain_core.tools import tool

from tool_eval.config import get_settings
from tool_eval.utils.paths import require_text_suffix, resolve_in_workspace, to_ref
from tool_eval.utils.text import append_text, read_text, truncate


@tool(parse_docstring=True)
def append_to_file(path: str, content: str) -> str:
    """Append text to a Markdown or text file in the workspace, creating it if needed.

    Args:
        path: Workspace-relative path ending in .md or .txt.
        content: Text to append.

    Returns:
        Confirmation with the file path.
    """
    settings = get_settings()
    target = resolve_in_workspace(path, workspace=settings.workspace_dir)
    require_text_suffix(target)
    append_text(target, content, max_bytes=settings.max_file_bytes)
    return (
        f"Appended {len(content)} characters to {to_ref(target, workspace=settings.workspace_dir)}"
    )


@tool(parse_docstring=True)
def read_file(path: str) -> str:
    """Read a Markdown or text file from the workspace.

    Args:
        path: Workspace-relative path ending in .md or .txt.

    Returns:
        The file contents, cut to the configured maximum length.
    """
    settings = get_settings()
    target = resolve_in_workspace(path, workspace=settings.workspace_dir)
    require_text_suffix(target)
    text = read_text(target, max_bytes=settings.max_file_bytes)
    return truncate(text, max_chars=settings.text_max_chars)
