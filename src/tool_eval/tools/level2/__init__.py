"""Level 2: medium tools that chain several operations."""

from langchain_core.tools import BaseTool

from tool_eval.tools.level2.bibliography import generate_bibliography
from tool_eval.tools.level2.papers import get_and_summarize_paper

TOOLS: list[BaseTool] = [get_and_summarize_paper, generate_bibliography]
