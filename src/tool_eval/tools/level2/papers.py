"""Level 2 tool: download, extract, split and summarize one paper in a single call."""

from langchain_core.tools import tool

from tool_eval.config import get_settings
from tool_eval.llm import get_chat_model
from tool_eval.services import papers


@tool(parse_docstring=True)
def get_and_summarize_paper(arxiv_id: str) -> str:
    """Download an arXiv paper, read its leading chunks and summarize it in a few sentences.

    Args:
        arxiv_id: arXiv identifier returned by a search, in the form <arxiv_id>.

    Returns:
        JSON with the summary text and the workspace path where it was saved.
    """
    result = papers.summarize_paper(arxiv_id, settings=get_settings(), llm=get_chat_model())
    return result.model_dump_json()
