"""Graph entry points for LangGraph Studio (`langgraph dev`); one agent per tool level."""

from tool_eval.agent import build_agent

level1 = build_agent("level1")
level2 = build_agent("level2")
level3 = build_agent("level3")
