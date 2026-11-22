"""Cost metering through LangChain callbacks, including LLM calls made inside tools."""

import re
import threading
import time
from collections.abc import Callable, Collection, Sequence
from typing import Any, Literal
from uuid import UUID

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tool_eval.evaluation.models import CostMetrics
from tool_eval.exceptions import ExternalServiceError

_Side = Literal["agent", "tool"]
# A tool call the server's parser did not recognise, left in the reply text: a JSON object with
# a name and its parameters/arguments, a <tool_call> block, or a <function=...> tag.
_TEXT_TOOL_CALL = re.compile(
    r'^\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"(?:parameters|arguments)"\s*:|<tool_call>|<function='
)


class RunMeter(BaseCallbackHandler):
    """Counts LLM calls, tokens, tool calls and time for one agent run.

    An LLM call whose parent run is a tool is attributed to that tool ("tool-internal");
    every other call belongs to the agent loop. Durations are summed per call, so calls that
    overlap in time (parallel tool calls) all count. Malformed tool calls never start a tool
    run, so they are counted where the agent's model returns them: calls whose arguments could
    not be parsed (`AIMessage.invalid_tool_calls`) and calls to a tool the level does not offer.
    """

    def __init__(
        self,
        *,
        tool_names: Collection[str] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Start a fresh meter and its wall clock.

        Args:
            tool_names: Names of the tools the agent may call. Without them, calls to unknown
                tools cannot be recognised and `unknown_tool_calls` stays unknown.
            clock: Monotonic clock in seconds, injectable for tests.
        """
        super().__init__()
        self._lock = threading.Lock()
        self._clock = clock
        self._started = clock()
        self._tool_names = frozenset(tool_names) if tool_names is not None else None
        self._tool_runs: set[UUID] = set()
        self._llm_inside_tool: dict[UUID, bool] = {}
        self._llm_started: dict[UUID, float] = {}
        self._tool_started: dict[UUID, float] = {}
        self._first_call_input_tokens: int | None = None
        self._counts: dict[str, int] = {
            "agent_llm_calls": 0,
            "tool_llm_calls": 0,
            "agent_input_tokens": 0,
            "agent_output_tokens": 0,
            "tool_input_tokens": 0,
            "tool_output_tokens": 0,
            "llm_calls_without_usage": 0,
            "tool_calls": 0,
            "tool_errors": 0,
        }
        self._invalid_tool_calls = 0
        self._unknown_tool_calls = 0
        self._text_tool_calls = 0
        self._environment_errors = 0
        self._seconds: dict[str, float] = {
            "agent_llm_seconds": 0.0,
            "tool_llm_seconds": 0.0,
            "tool_seconds": 0.0,
        }

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Remember when a model call started and whether a tool started it."""
        with self._lock:
            self._llm_inside_tool[run_id] = parent_run_id in self._tool_runs
            self._llm_started[run_id] = self._clock()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        """Add the call, its duration and its reported token usage to the right bucket."""
        with self._lock:
            side = self._end_llm_call(run_id)
            self._counts[f"{side}_llm_calls"] += 1
            replies = [
                generation.message
                for batch in response.generations
                for generation in batch
                if isinstance(generation, ChatGeneration)
                and isinstance(generation.message, AIMessage)
            ]
            usages = [reply.usage_metadata for reply in replies if reply.usage_metadata is not None]
            if not usages:
                self._counts["llm_calls_without_usage"] += 1
            for usage in usages:
                self._counts[f"{side}_input_tokens"] += usage["input_tokens"]
                self._counts[f"{side}_output_tokens"] += usage["output_tokens"]
            if side == "agent":
                if self._counts["agent_llm_calls"] == 1 and usages:
                    self._first_call_input_tokens = sum(u["input_tokens"] for u in usages)
                self._count_malformed(replies)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Add the duration of a failed model call; it does not count as a call."""
        with self._lock:
            self._end_llm_call(run_id)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Count a tool call and mark its run as a possible parent of model calls."""
        with self._lock:
            self._tool_runs.add(run_id)
            self._tool_started[run_id] = self._clock()
            self._counts["tool_calls"] += 1

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Add the duration of a finished tool call."""
        with self._lock:
            self._end_tool_call(run_id)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Count a failed tool call and add its duration."""
        with self._lock:
            self._counts["tool_errors"] += 1
            if _is_environment_error(error):
                self._environment_errors += 1
            self._end_tool_call(run_id)

    def snapshot(self) -> CostMetrics:
        """Return the metrics collected so far.

        Returns:
            Counters, summed call durations and the seconds elapsed since the meter was
            created.
        """
        with self._lock:
            return CostMetrics(
                **self._counts,
                **{name: round(seconds, 3) for name, seconds in self._seconds.items()},
                wall_seconds=round(self._clock() - self._started, 3),
                first_call_input_tokens=self._first_call_input_tokens,
                invalid_tool_calls=self._invalid_tool_calls,
                unknown_tool_calls=(
                    self._unknown_tool_calls if self._tool_names is not None else None
                ),
                text_tool_calls=self._text_tool_calls,
                tool_errors_environment=self._environment_errors,
                tool_errors_agent=self._counts["tool_errors"] - self._environment_errors,
            )

    def _end_llm_call(self, run_id: UUID) -> _Side:
        side: _Side = "tool" if self._llm_inside_tool.pop(run_id, False) else "agent"
        started = self._llm_started.pop(run_id, None)
        if started is not None:
            self._seconds[f"{side}_llm_seconds"] += self._clock() - started
        return side

    def _end_tool_call(self, run_id: UUID) -> None:
        started = self._tool_started.pop(run_id, None)
        if started is not None:
            self._seconds["tool_seconds"] += self._clock() - started

    def _count_malformed(self, replies: Sequence[AIMessage]) -> None:
        for reply in replies:
            self._invalid_tool_calls += len(reply.invalid_tool_calls)
            if (
                not reply.tool_calls
                and not reply.invalid_tool_calls
                and _TEXT_TOOL_CALL.search(reply.text)
            ):
                self._text_tool_calls += 1
            if self._tool_names is not None:
                self._unknown_tool_calls += sum(
                    call["name"] not in self._tool_names for call in reply.tool_calls
                )


def _is_environment_error(error: BaseException) -> bool:
    """Whether a failed tool call is caused by an external service rather than the agent."""
    seen: BaseException | None = error
    while seen is not None:
        if isinstance(seen, ExternalServiceError | httpx.HTTPError):
            return True
        seen = seen.__cause__
    return False
