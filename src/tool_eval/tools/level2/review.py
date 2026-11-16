"""Level 2 tool: write the review file."""

from langchain_core.tools import tool

from tool_eval.config import get_settings
from tool_eval.services import reports


@tool(parse_docstring=True)
def write_review(markdown: str) -> str:
    """Write the complete review to reviews/review.md, replacing whatever the file contained.

    Pass the whole review every time; to fix something, call it again with the corrected
    full text. Add the bibliography afterwards with append_bibliography.

    Args:
        markdown: The complete review in Markdown.

    Returns:
        Workspace-relative path of the review file.
    """
    return reports.write_review(markdown, settings=get_settings())
