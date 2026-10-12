"""Level 3: high-level tools that run a whole workflow from one request."""

from langchain_core.tools import BaseTool

from tool_eval.tools.level3.review import conduct_literature_review

TOOLS: list[BaseTool] = [conduct_literature_review]
