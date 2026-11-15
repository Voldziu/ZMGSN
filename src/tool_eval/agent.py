"""Agent construction: a tool-calling ReAct-style loop built with LangChain's `create_agent`."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest, ToolErrorMiddleware, wrap_tool_call
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from tool_eval.config import Settings, get_settings
from tool_eval.exceptions import ToolEvalError
from tool_eval.llm import build_chat_model
from tool_eval.prompts import load_prompt
from tool_eval.tools.arguments import coerce_json_string_args
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
        middleware=[decode_json_string_args, ToolErrorMiddleware(_report_domain_error)],
    )


@wrap_tool_call
def decode_json_string_args(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    """Decode list/object arguments sent as JSON strings before the tool validates them.

    Args:
        request: The pending tool call.
        handler: Executes the (possibly rewritten) call.

    Returns:
        The tool's result.
    """
    if request.tool is not None:
        args = request.tool_call["args"]
        decoded = coerce_json_string_args(args, request.tool.args)
        if decoded != args:
            request = request.override(tool_call={**request.tool_call, "args": decoded})
    return handler(request)


def run_task(
    agent: AgentGraph,
    task: str,
    *,
    max_steps: int,
    callbacks: Sequence[BaseCallbackHandler] = (),
    run_name: str | None = None,
    tags: Sequence[str] = (),
    metadata: Mapping[str, str | int | float] | None = None,
) -> list[AnyMessage]:
    """Run the agent on one task and return the full message history.

    Args:
        agent: Agent from `build_agent`.
        task: The user's request.
        max_steps: Maximum number of model calls.
        callbacks: Handlers that observe every model and tool call, including nested ones.
        run_name: Name of the trace in LangSmith when tracing is enabled.
        tags: Trace tags for filtering.
        metadata: Trace metadata, for example level and scenario.

    Returns:
        All messages of the run, including tool calls and results.

    Raises:
        GraphRecursionError: If the agent does not finish within `max_steps`.
    """
    # Each step is one model call plus one tool execution, so allow two graph steps per step.
    config: RunnableConfig = {
        "recursion_limit": 2 * max_steps + 1,
        "callbacks": list(callbacks),
        "tags": list(tags),
        "metadata": dict(metadata or {}),
    }
    if run_name:
        config["run_name"] = run_name
    result = agent.invoke({"messages": [HumanMessage(task)]}, config=config)
    messages: list[AnyMessage] = result["messages"]
    return messages


def _report_domain_error(exc: Exception, request: ToolCallRequest) -> str | None:
    if isinstance(exc, ToolEvalError):
        return f"{request.tool_call['name']} failed ({type(exc).__name__}): {exc}"
    return None
