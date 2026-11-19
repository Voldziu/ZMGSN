import functools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from test_evaluation import (
    _papers,
    _patch_agent,
    _record,
    _review_text,
    _run,
    _ToolCallingFake,
    _write_review,
)
from tool_eval.agent import run_task
from tool_eval.config import get_settings
from tool_eval.evaluation import runner
from tool_eval.evaluation.cost import RunMeter
from tool_eval.evaluation.models import CostMetrics, RunRecord
from tool_eval.evaluation.report import load_records


class _Clock:
    """A clock that only moves when told to."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _reply(
    content: str = "",
    *,
    input_tokens: int | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    invalid_tool_calls: list[dict[str, Any]] | None = None,
) -> LLMResult:
    usage = None
    if input_tokens is not None:
        usage = {"input_tokens": input_tokens, "output_tokens": 1, "total_tokens": input_tokens + 1}
    message = AIMessage(
        content,
        usage_metadata=usage,
        tool_calls=tool_calls or [],
        invalid_tool_calls=invalid_tool_calls or [],
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def _model_call(
    meter: RunMeter,
    clock: _Clock,
    seconds: float,
    reply: LLMResult,
    *,
    parent: UUID | None = None,
) -> None:
    run = uuid4()
    meter.on_chat_model_start({}, [[]], run_id=run, parent_run_id=parent)
    clock.advance(seconds)
    meter.on_llm_end(reply, run_id=run)


# --- older records ----------------------------------------------------------------


def test_records_without_new_measurements_read_them_as_unknown() -> None:
    old_cost = {
        "agent_llm_calls": 3,
        "tool_llm_calls": 2,
        "agent_input_tokens": 100,
        "agent_output_tokens": 20,
        "tool_input_tokens": 7,
        "tool_output_tokens": 3,
        "llm_calls_without_usage": 0,
        "tool_calls": 4,
        "tool_errors": 1,
        "wall_seconds": 12.5,
    }
    cost = CostMetrics.model_validate(old_cost)
    assert cost.total_tokens == 130
    assert cost.llm_calls == 5
    assert (cost.tool_calls, cost.tool_errors, cost.wall_seconds) == (4, 1, 12.5)
    for name in (
        "agent_llm_seconds",
        "tool_llm_seconds",
        "tool_seconds",
        "first_call_input_tokens",
        "invalid_tool_calls",
        "unknown_tool_calls",
    ):
        assert getattr(cost, name) is None, name
    assert cost.tool_other_seconds is None
    assert cost.malformed_tool_calls is None

    line = json.loads(_record("level1", "a", 1, score=1.0, tokens=100).model_dump_json())
    line["cost"] = old_cost
    record = RunRecord.model_validate(line)
    assert record.cost.agent_llm_seconds is None
    assert record.cost.total_tokens == 130


# --- time ---------------------------------------------------------------------------


def test_meter_sums_model_and_tool_time() -> None:
    clock = _Clock()
    meter = RunMeter(tool_names={"review"}, clock=clock)
    for _ in range(2):
        _model_call(meter, clock, 3, _reply(input_tokens=10))

    tool_run = uuid4()
    meter.on_tool_start({}, "", run_id=tool_run)
    clock.advance(10)
    for seconds in (15, 25):
        _model_call(meter, clock, seconds, _reply("summary", input_tokens=5), parent=tool_run)
    clock.advance(10)
    meter.on_tool_end("done", run_id=tool_run)

    cost = meter.snapshot()
    assert cost.agent_llm_seconds == 6
    assert cost.tool_llm_seconds == 40
    assert cost.tool_seconds == 60
    assert cost.tool_other_seconds == 20
    assert cost.wall_seconds == 66
    assert (cost.agent_llm_calls, cost.tool_llm_calls, cost.tool_calls) == (2, 2, 1)


def test_failed_calls_add_time_but_only_tools_count_as_errors() -> None:
    clock = _Clock()
    meter = RunMeter(clock=clock)
    run = uuid4()
    meter.on_chat_model_start({}, [[]], run_id=run)
    clock.advance(2)
    meter.on_llm_error(TimeoutError("slow"), run_id=run)

    tool_run = uuid4()
    meter.on_tool_start({}, "", run_id=tool_run)
    clock.advance(4)
    meter.on_tool_error(ValueError("bad input"), run_id=tool_run)

    cost = meter.snapshot()
    assert (cost.agent_llm_seconds, cost.tool_seconds) == (2, 4)
    assert cost.agent_llm_calls == 0
    assert (cost.tool_calls, cost.tool_errors) == (1, 1)


def test_overlapping_tool_calls_are_all_counted() -> None:
    clock = _Clock()
    meter = RunMeter(clock=clock)
    first, second = uuid4(), uuid4()
    meter.on_tool_start({}, "", run_id=first)
    meter.on_tool_start({}, "", run_id=second)
    clock.advance(5)
    meter.on_tool_end("a", run_id=first)
    meter.on_tool_end("b", run_id=second)
    cost = meter.snapshot()
    assert cost.tool_seconds == 10
    assert cost.wall_seconds == 5


# --- prompt overhead -------------------------------------------------------------------


def test_first_call_input_tokens_ignore_later_and_tool_internal_calls() -> None:
    clock = _Clock()
    meter = RunMeter(clock=clock)
    tool_run = uuid4()
    meter.on_tool_start({}, "", run_id=tool_run)
    _model_call(meter, clock, 1, _reply(input_tokens=7), parent=tool_run)
    meter.on_tool_end("done", run_id=tool_run)
    _model_call(meter, clock, 1, _reply(input_tokens=100))
    _model_call(meter, clock, 1, _reply(input_tokens=150))
    assert meter.snapshot().first_call_input_tokens == 100


def test_first_call_without_usage_leaves_the_overhead_unknown() -> None:
    clock = _Clock()
    meter = RunMeter(clock=clock)
    _model_call(meter, clock, 1, _reply())
    _model_call(meter, clock, 1, _reply(input_tokens=150))
    cost = meter.snapshot()
    assert cost.first_call_input_tokens is None
    assert cost.llm_calls_without_usage == 1


# --- malformed tool calls ---------------------------------------------------------------


@tool
def probe() -> str:
    """Return a fixed answer."""
    return "ok"


@tool
def count_words(text: str, limit: int) -> int:
    """Count the words of a text, up to a limit."""
    return min(len(text.split()), limit)


def test_malformed_tool_calls_are_counted_but_never_run() -> None:
    model = _ToolCallingFake(
        messages=iter(
            [
                AIMessage(
                    "",
                    tool_calls=[
                        {"name": "search_arxiv", "args": {}, "id": "call-1"},
                        {"name": "probe", "args": {}, "id": "call-2"},
                    ],
                    invalid_tool_calls=[
                        {
                            "name": "probe",
                            "args": '{"broken": ',
                            "id": "call-3",
                            "error": "not valid JSON",
                            "type": "invalid_tool_call",
                        }
                    ],
                ),
                AIMessage("done"),
            ]
        )
    )
    meter = RunMeter(tool_names={"probe"})
    config: RunnableConfig = {"callbacks": [meter]}
    result = create_agent(model=model, tools=[probe]).invoke(
        {"messages": [HumanMessage("go")]}, config=config
    )

    answers = {m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)}
    assert answers["call-1"].status == "error"
    assert "not a valid tool" in str(answers["call-1"].content)
    assert answers["call-2"].content == "ok"
    cost = meter.snapshot()
    assert (cost.invalid_tool_calls, cost.unknown_tool_calls) == (1, 1)
    assert cost.malformed_tool_calls == 2
    assert (cost.tool_calls, cost.tool_errors) == (1, 0)


def test_malformed_calls_survive_the_step_limit() -> None:
    model = _ToolCallingFake(
        messages=iter(
            AIMessage("", tool_calls=[{"name": "search_arxiv", "args": {}, "id": f"call-{i}"}])
            for i in range(20)
        )
    )
    meter = RunMeter(tool_names={"probe"})
    with pytest.raises(GraphRecursionError):
        run_task(create_agent(model=model, tools=[probe]), "go", max_steps=3, callbacks=[meter])
    cost = meter.snapshot()
    assert cost.agent_llm_calls >= 3
    assert cost.unknown_tool_calls == cost.agent_llm_calls
    assert cost.tool_calls == 0


def test_meter_without_tool_names_cannot_recognise_unknown_tools() -> None:
    clock = _Clock()
    meter = RunMeter(clock=clock)
    call = {"name": "anything", "args": {}, "id": "call-1"}
    _model_call(meter, clock, 1, _reply(tool_calls=[call]))
    cost = meter.snapshot()
    assert cost.invalid_tool_calls == 0
    assert cost.unknown_tool_calls is None
    assert cost.malformed_tool_calls is None


def test_tool_internal_replies_are_not_checked_for_tool_calls() -> None:
    clock = _Clock()
    meter = RunMeter(tool_names={"probe"}, clock=clock)
    tool_run = uuid4()
    meter.on_tool_start({}, "", run_id=tool_run)
    call = {"name": "elsewhere", "args": {}, "id": "call-1"}
    _model_call(meter, clock, 1, _reply(tool_calls=[call]), parent=tool_run)
    assert meter.snapshot().unknown_tool_calls == 0


# --- tool errors -------------------------------------------------------------------------


def test_validation_failure_counts_one_tool_call_and_one_error() -> None:
    model = _ToolCallingFake(
        messages=iter(
            [
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "count_words",
                            "args": {"text": "a b c", "limit": "many"},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage("done"),
            ]
        )
    )
    meter = RunMeter(tool_names={"count_words"})
    config: RunnableConfig = {"callbacks": [meter]}
    result = create_agent(model=model, tools=[count_words]).invoke(
        {"messages": [HumanMessage("go")]}, config=config
    )
    [answer] = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert answer.status == "error"
    assert result["messages"][-1].content == "done"  # the agent continued
    cost = meter.snapshot()
    assert (cost.tool_calls, cost.tool_errors) == (1, 1)
    assert cost.malformed_tool_calls == 0


# --- runner ----------------------------------------------------------------------------


def test_wall_time_ends_before_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    monkeypatch.setattr(runner, "RunMeter", functools.partial(RunMeter, clock=clock))

    def agent_run() -> None:
        clock.advance(5)
        _write_review(get_settings().workspace_dir, _review_text())

    papers = _papers()

    def throttled_lookup(ids: Sequence[str]) -> Mapping[str, Any]:
        clock.advance(30)  # arXiv lookups while the review is scored
        return {i: papers.get(i) for i in ids}

    _patch_agent(monkeypatch, agent_run)
    _, output = _run(tmp_path, lookup=throttled_lookup)
    records = load_records(output)
    assert len(records) == 2
    for record in records:
        assert record.success
        assert record.cost.wall_seconds == 5
        assert (record.cost.agent_llm_seconds, record.cost.tool_seconds) == (0, 0)
        # The runner gives the meter the level's tool names.
        assert record.cost.malformed_tool_calls == 0
