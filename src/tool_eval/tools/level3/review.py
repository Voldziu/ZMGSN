"""Level 3 tool: complete literature review from a single query."""

from langchain_core.tools import tool

from tool_eval.config import get_settings
from tool_eval.llm import get_chat_model
from tool_eval.services import reports


@tool(parse_docstring=True)
def conduct_literature_review(query: str) -> str:
    """Write the complete literature review on a topic in one call.

    Searches arXiv within the task's date window, reads the full text of the required number of
    papers (replacing papers that cannot be read), summarizes each paper, and saves
    reviews/review.md in the structure the task requires: a heading with the topic, one section
    per paper with its APA reference and summary, and a bibliography.

    Args:
        query: Research topic.

    Returns:
        JSON with a message saying whether the review is complete, the review path, how many
        papers were requested and read in full, and the candidates skipped with the reason.
    """
    result = reports.conduct_review(query, settings=get_settings(), llm=get_chat_model())
    return result.model_dump_json()
