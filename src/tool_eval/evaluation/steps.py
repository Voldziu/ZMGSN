"""One short log line per agent step, and warnings when the agent seems to be stuck in a loop.

`StepLogger` is a LangChain callback handler that the runner passes next to `RunMeter`. It writes
each line as soon as its event ends, so runs that hit the step limit or are stopped keep the steps
they took. Arguments, results and text are cut short; the full messages stay in MLflow traces.
The loop warnings only observe: they never stop a run or change its record.
"""

import json
import re
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tool_eval.utils.logger import get_logger

logger = get_logger(__name__)

MAX_PART_CHARS = 150
REPEATED_CALL_WARNING = 3  # the same tool call this many times in a row
ERROR_STREAK_WARNING = 5  # this many failed tool calls in a row
_NORMAL_FINISH = frozenset({"stop", "tool_calls"})
_WHITESPACE = re.compile(r"\s+")


def shorten(value: object, limit: int = MAX_PART_CHARS) -> str:
    """Render a value on one line and cut it to at most `limit` characters.

    Args:
        value: Text, or any value, which is rendered with `str`.
        limit: Maximum length of the result, including the `...` that marks a cut.

    Returns:
        The text with every run of whitespace collapsed to one space, cut if it is too long.
    """
    text = _WHITESPACE.sub(" ", value if isinstance(value, str) else str(value)).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class _ToolCall:
    """A tool call that has started and not ended yet."""

    name: str
    arguments: str
    step: int
    started: float
    model_calls: int = 0
    model_seconds: float = 0.0


class StepLogger(BaseCallbackHandler):
    """Logs one DEBUG line per reply of the agent's model and per tool call, and warns of loops.

    A model call whose parent run is a tool run belongs to that tool: it gets no line of its own,
    and the tool's line counts it. Every other model call is a reply of the agent and starts the
    next step. A tool call carries the step of the reply that requested it.

    Two signals are watched, each warned about at most once per run: the same tool call (tool and
    arguments) for the `REPEATED_CALL_WARNING`-th time in a row, and `ERROR_STREAK_WARNING`
    failed tool calls in a row.
    """

    def __init__(
        self,
        *,
        tool_names: Collection[str] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Start a run's step log.

        Args:
            tool_names: Names of the tools the agent may call; a reply that calls another name
                counts it as malformed. Without them, only unparseable calls count.
            clock: Monotonic clock in seconds, injectable for tests.
        """
        super().__init__()
        self._lock = threading.Lock()
        self._clock = clock
        self._tool_names = frozenset(tool_names) if tool_names is not None else None
        self._tools: dict[UUID, _ToolCall] = {}
        self._model_calls: dict[UUID, tuple[UUID | None, float]] = {}
        self._step = 0
        self._last_call: str | None = None
        self._same_calls = 0
        self._failed_calls = 0
        self._warned_repeat = False
        self._warned_errors = False

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Remember when a model call started and which tool, if any, made it."""
        with self._lock:
            tool_run = parent_run_id if parent_run_id in self._tools else None
            self._model_calls[run_id] = (tool_run, self._clock())

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        """Log a reply of the agent, or count a model call on the tool that made it."""
        with self._lock:
            tool_run, seconds = self._end_model_call(run_id, counted=True)
            if tool_run is not None:
                return
            self._step += 1
            replies = [
                generation.message
                for batch in response.generations
                for generation in batch
                if isinstance(generation, ChatGeneration)
                and isinstance(generation.message, AIMessage)
            ]
            logger.debug("#%d model -> %s", self._step, self._describe(replies, seconds))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Forget a failed model call; one inside a tool adds its time to that tool."""
        with self._lock:
            self._end_model_call(run_id, counted=False)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Remember a tool call and warn when the agent repeats the same call."""
        name = str((serialized or {}).get("name") or kwargs.get("name") or "tool")
        arguments: Mapping[str, Any] | str = inputs if inputs is not None else input_str
        key = json.dumps([name, arguments], sort_keys=True, default=str)
        with self._lock:
            call = _ToolCall(
                name=name,
                arguments=_render(arguments),
                step=self._step,
                started=self._clock(),
            )
            self._tools[run_id] = call
            if key == self._last_call:
                self._same_calls += 1
            else:
                self._last_call, self._same_calls = key, 1
            if self._same_calls >= REPEATED_CALL_WARNING and not self._warned_repeat:
                self._warned_repeat = True
                logger.warning(
                    "same tool call %d times in a row: %s(%s)",
                    self._same_calls,
                    call.name,
                    call.arguments,
                )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Log a tool call that returned a result."""
        result = output.content if isinstance(output, ToolMessage) else output
        with self._lock:
            self._failed_calls = 0
            call = self._tools.pop(run_id, None)
            if call is not None:
                logger.debug(
                    "#%d %s(%s) -> ok, %s: %s",
                    call.step,
                    call.name,
                    call.arguments,
                    self._timing(call),
                    shorten(result),
                )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Log a failed tool call and warn when tool calls keep failing."""
        with self._lock:
            self._failed_calls += 1
            call = self._tools.pop(run_id, None)
            name = call.name if call is not None else "tool"
            if call is not None:
                logger.debug(
                    "#%d %s(%s) -> error, %s: %s",
                    call.step,
                    call.name,
                    call.arguments,
                    self._timing(call),
                    shorten(f"{type(error).__name__}: {error}"),
                )
            if self._failed_calls >= ERROR_STREAK_WARNING and not self._warned_errors:
                self._warned_errors = True
                logger.warning(
                    "%d tool calls in a row failed; the last one: %s", self._failed_calls, name
                )

    def _end_model_call(self, run_id: UUID, *, counted: bool) -> tuple[UUID | None, float]:
        tool_run, started = self._model_calls.pop(run_id, (None, None))
        seconds = self._clock() - started if started is not None else 0.0
        call = self._tools.get(tool_run) if tool_run is not None else None
        if call is not None:
            call.model_calls += int(counted)
            call.model_seconds += seconds
        return tool_run, seconds

    def _describe(self, replies: Sequence[AIMessage], seconds: float) -> str:
        calls = sum(len(reply.tool_calls) for reply in replies)
        malformed = sum(len(reply.invalid_tool_calls) for reply in replies)
        if self._tool_names is not None:
            malformed += sum(
                call["name"] not in self._tool_names
                for reply in replies
                for call in reply.tool_calls
            )
        parts = [f"{calls} tool call{'' if calls == 1 else 's'}"]
        if malformed:
            parts.append(f"{malformed} malformed")
        finishes = {str(reply.response_metadata.get("finish_reason") or "") for reply in replies}
        parts.extend(f"finish={reason}" for reason in sorted(finishes - {"", *_NORMAL_FINISH}))
        parts.append(f"{seconds:.1f} s")
        text = shorten(" ".join(reply.text for reply in replies))
        if text:
            parts.append(f"text: {text}")
        return ", ".join(parts)

    def _timing(self, call: _ToolCall) -> str:
        timing = f"{self._clock() - call.started:.1f} s"
        if call.model_calls:
            timing += f", {call.model_calls} model calls in {call.model_seconds:.1f} s"
        return timing


def _render(arguments: Mapping[str, Any] | str) -> str:
    if isinstance(arguments, str):
        return shorten(arguments)
    return shorten(", ".join(f"{key}={value!r}" for key, value in arguments.items()))
