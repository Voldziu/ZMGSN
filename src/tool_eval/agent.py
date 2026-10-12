"""Agent construction: a tool-calling ReAct-style loop built with LangChain's `create_agent`."""

from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest, ToolErrorMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from tool_eval.config import Settings, get_settings
from tool_eval.exceptions import ToolEvalError
from tool_eval.llm import build_chat_model
from tool_eval.prompts import load_prompt
from tool_eval.tools.registry import get_tools

AgentGraph = CompiledStateGraph[Any, Any, Any, Any]


def build_agent(level: str, *, settings: Settings | None = None) -> AgentGraph:
    """Build an agent that sees exactly one level of tools.

    Args:
        level: Tool level key, for example `level1`.
        settings: Application settings; defaults to the cached `.env` settings.

    Returns:
        The compiled agent graph.
    """
    active = settings or get_settings()
    return create_agent(
        model=build_chat_model(active),
        tools=get_tools(level),
        system_prompt=load_prompt("system", active.prompt_version).text,
        middleware=[ToolErrorMiddleware(_report_domain_error)],
    )


def run_task(
    agent: AgentGraph,
    task: str,
    *,
    max_steps: int,
    callbacks: Sequence[BaseCallbackHandler] = (),
) -> list[AnyMessage]:
    """Run the agent on one task and return the full message history.

    Args:
        agent: Agent from `build_agent`.
        task: The user's request.
        max_steps: Maximum number of model calls.
        callbacks: Handlers that observe every model and tool call, including nested ones.

    Returns:
        All messages of the run, including tool calls and results.

    Raises:
        GraphRecursionError: If the agent does not finish within `max_steps`.
    """
    # Each step is one model call plus one tool execution, so allow two graph steps per step.
    config: RunnableConfig = {"recursion_limit": 2 * max_steps + 1, "callbacks": list(callbacks)}
    result = agent.invoke({"messages": [HumanMessage(task)]}, config=config)
    messages: list[AnyMessage] = result["messages"]
    return messages


def _report_domain_error(exc: Exception, request: ToolCallRequest) -> str | None:
    if isinstance(exc, ToolEvalError):
        return f"{request.tool_call['name']} failed ({type(exc).__name__}): {exc}"
    return None
