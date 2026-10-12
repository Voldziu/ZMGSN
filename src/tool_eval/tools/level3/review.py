"""Level 3 tool: complete literature review from a single query."""

from langchain_core.tools import tool

from tool_eval.config import get_settings
from tool_eval.llm import get_chat_model
from tool_eval.services import reports


@tool(parse_docstring=True)
def conduct_literature_review(query: str) -> str:
    """Find recent arXiv papers on a topic, summarize them and save a Markdown review.

    Args:
        query: Research topic.

    Returns:
        JSON with the workspace path of the review file and the number of papers included.
    """
    result = reports.conduct_review(query, settings=get_settings(), llm=get_chat_model())
    return result.model_dump_json()
