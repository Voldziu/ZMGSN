"""Cost metering through LangChain callbacks, including LLM calls made inside tools."""

import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tool_eval.evaluation.models import CostMetrics


class RunMeter(BaseCallbackHandler):
    """Counts LLM calls, tokens and tool calls for one agent run.

    An LLM call whose parent run is a tool is attributed to that tool ("tool-internal");
    every other call belongs to the agent loop.
    """

    def __init__(self) -> None:
        """Start a fresh meter and its wall clock."""
        super().__init__()
        self._lock = threading.Lock()
        self._started = time.perf_counter()
        self._tool_runs: set[UUID] = set()
        self._llm_inside_tool: dict[UUID, bool] = {}
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

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Remember whether a model call was started by a tool."""
        with self._lock:
            self._llm_inside_tool[run_id] = parent_run_id in self._tool_runs

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        """Add the call and its reported token usage to the right bucket."""
        with self._lock:
            side = "tool" if self._llm_inside_tool.pop(run_id, False) else "agent"
            self._counts[f"{side}_llm_calls"] += 1
            usages = [
                generation.message.usage_metadata
                for batch in response.generations
                for generation in batch
                if isinstance(generation, ChatGeneration)
                and isinstance(generation.message, AIMessage)
                and generation.message.usage_metadata is not None
            ]
            if not usages:
                self._counts["llm_calls_without_usage"] += 1
            for usage in usages:
                self._counts[f"{side}_input_tokens"] += usage["input_tokens"]
                self._counts[f"{side}_output_tokens"] += usage["output_tokens"]

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Forget a failed model call."""
        with self._lock:
            self._llm_inside_tool.pop(run_id, None)

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
            self._counts["tool_calls"] += 1

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Count a failed tool call."""
        with self._lock:
            self._counts["tool_errors"] += 1

    def snapshot(self) -> CostMetrics:
        """Return the metrics collected so far.

        Returns:
            Counters plus the seconds elapsed since the meter was created.
        """
        with self._lock:
            return CostMetrics(
                **self._counts,
                wall_seconds=round(time.perf_counter() - self._started, 3),
            )
