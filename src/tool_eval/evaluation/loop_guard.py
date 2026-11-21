"""Stop a run as soon as its agent is clearly stuck in a loop.

`LoopGuard` is a LangChain callback handler that the runner passes last, after `RunMeter` and
`StepLogger`. Unlike them it changes the run: it sets `raise_error`, so the `AgentLoopError` it
raises leaves the agent loop and the runner records the run as `loop`. Two signals count, both on
the agent's own tool calls (model calls made inside a tool never count):

- the same tool call (tool and arguments) requested `max_repeated_calls` times in a row: the run
  stops when the reply that requests it ends, before the call executes;
- `max_tool_error_streak` failed tool calls in a row: the run stops when the agent's model is
  about to be called again.

A limit of 0 switches its check off.
"""

import json
import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tool_eval.evaluation.steps import shorten
from tool_eval.exceptions import AgentLoopError


class LoopGuard(BaseCallbackHandler):
    """Raises `AgentLoopError` when the agent repeats one tool call or tool calls keep failing.

    A model call that starts while no tool run is active is a reply of the agent; any other
    model call belongs to a tool and is ignored. This also holds for model calls nested several
    runs below the tool run.
    """

    raise_error = True

    def __init__(self, *, max_repeated_calls: int, max_tool_error_streak: int) -> None:
        """Start a run's guard.

        Args:
            max_repeated_calls: Identical tool calls in a row that stop the run; 0 disables.
            max_tool_error_streak: Failed tool calls in a row that stop the run; 0 disables.
        """
        super().__init__()
        self._max_repeated = max_repeated_calls
        self._max_errors = max_tool_error_streak
        self._lock = threading.Lock()
        self._active_tools: dict[UUID, str] = {}
        self._agent_calls: set[UUID] = set()
        self._last_call: str | None = None
        self._same_calls = 0
        self._failed_calls = 0
        self._last_failed_tool = "tool"

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Stop the run before the agent's model is called again after an error streak."""
        with self._lock:
            if self._active_tools:
                return
            self._agent_calls.add(run_id)
            if self._max_errors and self._failed_calls >= self._max_errors:
                raise AgentLoopError(
                    f"{self._failed_calls} tool calls in a row failed; "
                    f"the last one: {self._last_failed_tool}"
                )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        """Count the tool calls an agent reply requests and stop at a repeated one."""
        with self._lock:
            if run_id not in self._agent_calls:
                return
            self._agent_calls.discard(run_id)
            for batch in response.generations:
                for generation in batch:
                    if not isinstance(generation, ChatGeneration):
                        continue
                    message = generation.message
                    if not isinstance(message, AIMessage):
                        continue
                    for call in message.tool_calls:
                        self._count_call(call["name"], call["args"])

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Forget a failed model call."""
        with self._lock:
            self._agent_calls.discard(run_id)

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        """Mark a tool run as active, so model calls inside it are not the agent's."""
        name = str((serialized or {}).get("name") or kwargs.get("name") or "tool")
        with self._lock:
            self._active_tools[run_id] = name

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """A tool call succeeded: end the error streak."""
        with self._lock:
            self._active_tools.pop(run_id, None)
            self._failed_calls = 0

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """A tool call failed: extend the error streak."""
        with self._lock:
            self._failed_calls += 1
            self._last_failed_tool = self._active_tools.pop(run_id, "tool")

    def _count_call(self, name: str, arguments: Any) -> None:
        key = json.dumps([name, arguments], sort_keys=True, default=str)
        if key == self._last_call:
            self._same_calls += 1
        else:
            self._last_call, self._same_calls = key, 1
        if self._max_repeated and self._same_calls >= self._max_repeated:
            rendered = shorten(json.dumps(arguments, sort_keys=True, default=str))
            raise AgentLoopError(
                f"same tool call requested {self._same_calls} times in a row: {name}({rendered})"
            )
