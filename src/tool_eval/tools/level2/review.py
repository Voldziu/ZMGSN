"""Level 2 tool: append Markdown to the review file."""

from langchain_core.tools import tool

from tool_eval.config import get_settings
from tool_eval.services import reports


@tool(parse_docstring=True)
def append_to_review(markdown: str) -> str:
    """Append Markdown to the review file reviews/review.md, creating it if needed.

    Args:
        markdown: Text to append.

    Returns:
        Workspace-relative path of the review file.
    """
    return reports.append_to_review(markdown, settings=get_settings())
