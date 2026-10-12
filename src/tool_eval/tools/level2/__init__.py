"""Level 2: medium tools that chain several operations."""

from langchain_core.tools import BaseTool

from tool_eval.tools.level2.bibliography import append_bibliography
from tool_eval.tools.level2.papers import get_and_summarize_paper
from tool_eval.tools.level2.review import append_to_review
from tool_eval.tools.level2.search import search_papers

TOOLS: list[BaseTool] = [
    search_papers,
    get_and_summarize_paper,
    append_to_review,
    append_bibliography,
]
