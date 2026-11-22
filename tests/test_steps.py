import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import tool

from test_evaluation import SCENARIO, _lookup, _papers, _settings, _ToolCallingFake
from test_metering import _Clock
from tool_eval.evaluation import runner
from tool_eval.evaluation.models import RunRecord
from tool_eval.evaluation.report import load_records
from tool_eval.evaluation.steps import StepLogger
from tool_eval.exceptions import InvalidInputError
from tool_eval.utils.logger import close_logging, configure_logging

STEPS_LOGGER = "tool_eval.evaluation.steps"
TOOLS = {"search_arxiv", "read_file", "append_to_file", "conduct_literature_review"}


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def steps(clock: _Clock, caplog: pytest.LogCaptureFixture) -> StepLogger:
    caplog.set_level(logging.DEBUG, logger=STEPS_LOGGER)
    return StepLogger(tool_names=TOOLS, clock=clock)


def _reply(
    content: str = "",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    invalid_tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> LLMResult:
    message = AIMessage(
        content,
        tool_calls=tool_calls or [],
        invalid_tool_calls=invalid_tool_calls or [],
        response_metadata={"finish_reason": finish_reason} if finish_reason else {},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def _model_call(
    steps: StepLogger,
    clock: _Clock,
    seconds: float,
    reply: LLMResult,
    *,
    parent: UUID | None = None,
) -> None:
    run = uuid4()
    steps.on_chat_model_start({}, [[]], run_id=run, parent_run_id=parent)
    clock.advance(seconds)
    steps.on_llm_end(reply, run_id=run)


def _tool_call(
    steps: StepLogger,
    clock: _Clock,
    name: str,
    arguments: dict[str, Any],
    *,
    seconds: float = 1.0,
    result: str = "ok",
    error: Exception | None = None,
) -> None:
    run = uuid4()
    steps.on_tool_start({"name": name}, str(arguments), run_id=run, inputs=arguments)
    clock.advance(seconds)
    if error is not None:
        steps.on_tool_error(error, run_id=run)
    else:
        steps.on_tool_end(ToolMessage(result, tool_call_id="call"), run_id=run)


def _messages(caplog: pytest.LogCaptureFixture, level: int = logging.DEBUG) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == STEPS_LOGGER and r.levelno == level]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return _messages(caplog, logging.WARNING)


# --- step lines --------------------------------------------------------------------------


def test_a_reply_line_counts_tool_calls_and_malformed_calls(
    steps: StepLogger, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    calls = [
        {"name": "search_arxiv", "args": {"query": "rag"}, "id": "call-1"},
        {"name": "search_web", "args": {}, "id": "call-2"},
    ]
    broken = [
        {
            "name": "read_file",
            "args": '{"path": ',
            "id": "call-3",
            "error": "not valid JSON",
            "type": "invalid_tool_call",
        }
    ]
    _model_call(steps, clock, 2, _reply(tool_calls=calls, invalid_tool_calls=broken))
    assert _messages(caplog) == ["#1 model -> 2 tool calls, 2 malformed, 2.0 s"]


def test_a_reply_line_shows_unusual_finish_reasons_and_the_start_of_the_text(
    steps: StepLogger, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    _model_call(steps, clock, 1, _reply("The review is\nwritten.", finish_reason="stop"))
    _model_call(steps, clock, 1, _reply("x" * 400, finish_reason="length"))
    call = {"name": "read_file", "args": {"path": "a.md"}, "id": "call-1"}
    _model_call(steps, clock, 1, _reply(tool_calls=[call], finish_reason="tool_calls"))
    first, second, third = _messages(caplog)
    assert first == "#1 model -> 0 tool calls, 1.0 s, text: The review is written."
    assert second.startswith("#2 model -> 0 tool calls, finish=length, 1.0 s, text: xxx")
    assert second.endswith("x...")
    assert third == "#3 model -> 1 tool call, 1.0 s"


def test_a_failed_tool_call_shows_its_arguments_and_the_error(
    steps: StepLogger, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    _model_call(steps, clock, 1, _reply(tool_calls=[]))
    error = InvalidInputError("No papers found for the query.")
    _tool_call(
        steps, clock, "conduct_literature_review", {"query": "Reid"}, seconds=0.4, error=error
    )
    assert _messages(caplog)[-1] == (
        "#1 conduct_literature_review(query='Reid') -> error, 0.4 s: "
        "InvalidInputError: No papers found for the query."
    )


def test_a_successful_tool_call_shows_the_start_of_its_result(
    steps: StepLogger, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    _model_call(steps, clock, 1, _reply())
    _tool_call(steps, clock, "read_file", {"path": "reviews/review.md"}, result="# Review\n\nText")
    assert _messages(caplog)[-1] == (
        "#1 read_file(path='reviews/review.md') -> ok, 1.0 s: # Review Text"
    )


def test_long_arguments_and_results_are_cut_and_lines_never_break(
    steps: StepLogger, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    text = "word\n" * 1000  # 5,000 characters of review text
    _model_call(steps, clock, 1, _reply())
    _tool_call(steps, clock, "append_to_file", {"path": "r.md", "content": text}, result=text)
    line = _messages(caplog)[-1]
    arguments = line.split("append_to_file(", 1)[1].split(") -> ok", 1)[0]
    result = line.split(" s: ", 1)[1]
    assert len(arguments) == 150
    assert arguments.endswith("...")
    assert len(result) == 150
    assert result.endswith("...")
    assert all("\n" not in message for message in _messages(caplog))


def test_model_calls_inside_a_tool_are_summed_on_its_line(
    steps: StepLogger, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    _model_call(steps, clock, 1, _reply())
    tool_run = uuid4()
    steps.on_tool_start(
        {"name": "conduct_literature_review"}, "", run_id=tool_run, inputs={"query": "rag"}
    )
    _model_call(steps, clock, 3, _reply("summary"), parent=tool_run)
    _model_call(steps, clock, 4, _reply("summary"), parent=tool_run)
    clock.advance(3)
    steps.on_tool_end(ToolMessage("reviews/review.md", tool_call_id="call"), run_id=tool_run)
    assert _messages(caplog) == [
        "#1 model -> 0 tool calls, 1.0 s",
        "#1 conduct_literature_review(query='rag') -> ok, 10.0 s, "
        "2 model calls in 7.0 s: reviews/review.md",
    ]


# --- loop warnings -----------------------------------------------------------------------


def test_the_same_call_three_times_in_a_row_warns_once(
    steps: StepLogger, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    for _ in range(2):
        _tool_call(steps, clock, "conduct_literature_review", {"query": "rag"})
    assert _warnings(caplog) == []
    for _ in range(8):
        _tool_call(steps, clock, "conduct_literature_review", {"query": "rag"})
    assert _warnings(caplog) == [
        "same tool call 3 times in a row: conduct_literature_review(query='rag')"
    ]


def test_different_arguments_are_not_a_repeated_call(
    steps: StepLogger, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    for query in ("rag", "retrieval", "rag survey"):
        _tool_call(steps, clock, "search_arxiv", {"query": query})
    assert _warnings(caplog) == []


def test_arguments_that_differ_after_the_cut_are_not_identical(
    steps: StepLogger, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    for i in range(3):
        _tool_call(steps, clock, "append_to_file", {"content": "a" * 200 + str(i)})
    shown = {line.split(" -> ")[0] for line in _messages(caplog)}
    assert len(shown) == 1  # the shortened calls look the same
    assert _warnings(caplog) == []


def test_five_failed_calls_in_a_row_warn_once(
    steps: StepLogger, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    error = InvalidInputError("No papers found for the query.")
    for i in range(9):
        _tool_call(steps, clock, "conduct_literature_review", {"query": f"Reid {i}"}, error=error)
        assert len(_warnings(caplog)) == (1 if i >= 4 else 0)
    assert _warnings(caplog) == [
        "5 tool calls in a row failed; the last one: conduct_literature_review"
    ]


def test_a_successful_call_resets_the_failure_count(
    steps: StepLogger, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    error = InvalidInputError("No papers found for the query.")
    for i in range(4):
        _tool_call(steps, clock, "search_arxiv", {"query": f"a{i}"}, error=error)
    _tool_call(steps, clock, "search_arxiv", {"query": "b"})
    for i in range(4):
        _tool_call(steps, clock, "search_arxiv", {"query": f"c{i}"}, error=error)
    assert _warnings(caplog) == []


# --- in the runner ------------------------------------------------------------------------


@tool
def read_file(path: str) -> str:
    """Read a file from the workspace."""
    return f"content of {path}"


def _build_agent(replies: Sequence[AIMessage]) -> Callable[..., Any]:
    """A `build_agent` stand-in: a real agent loop with a scripted model and `read_file`."""

    def build_agent(level: str, *, settings: Any = None) -> Any:
        return create_agent(model=_ToolCallingFake(messages=iter(replies)), tools=[read_file])

    return build_agent


def _logged_run(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    replies: Sequence[AIMessage],
    **settings: Any,
) -> tuple[RunRecord, list[str]]:
    """Run one level-1 evaluation run and return its record and its own log lines."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(runner, "build_agent", _build_agent(replies))
    output, log_file = directory / "results.jsonl", directory / "results.log"
    configure_logging(log_file)
    try:
        runner.run_evaluation(
            scenarios=[SCENARIO],
            levels=["level1"],
            repeats=1,
            output=output,
            runs_dir=directory / "runs",
            settings=_settings(directory / "unused", llm_model="qwen", **settings),
            lookup=_lookup(_papers()),
            order_seed=7,
        )
    finally:
        close_logging()
    [record] = load_records(output)
    tag = f"[{record.run_id}] "
    lines = log_file.read_text(encoding="utf-8").splitlines()
    return record, [line.split(tag, 1)[1] for line in lines if tag in line]


def _read(i: int, path: str = "reviews/review.md") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": "read_file", "args": {"path": path}, "id": f"c{i}"}])


def test_step_lines_appear_in_order_between_the_run_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, lines = _logged_run(tmp_path, monkeypatch, [_read(1), AIMessage("done")])
    assert record.status == "completed"
    assert lines[0].startswith("start: position ")
    assert lines[1].startswith("#1 model -> 1 tool call, ")
    assert lines[2].startswith("#1 read_file(path='reviews/review.md') -> ok, ")
    assert lines[2].endswith(": content of reviews/review.md")
    assert lines[3].startswith("#2 model -> 0 tool calls, ")
    assert lines[3].endswith("text: done")
    assert lines[-1].startswith("end: completed, ")


def test_a_run_stopped_by_the_step_limit_keeps_its_step_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = [_read(i, f"notes/{i}.md") for i in range(10)]
    record, lines = _logged_run(tmp_path, monkeypatch, replies, agent_max_steps=2)
    assert record.status == "step_limit"
    assert "#1 read_file(path='notes/0.md') -> ok" in "\n".join(lines)
    assert "#2 read_file(path='notes/1.md') -> ok" in "\n".join(lines)
    assert lines[-1].startswith("end: step_limit, success=False, ")


def test_a_failing_step_logger_never_changes_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = [_read(1), AIMessage("done")]
    baseline, _ = _logged_run(tmp_path / "a", monkeypatch, replies)

    def broken(self: StepLogger, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("log disk full")

    monkeypatch.setattr(StepLogger, "on_tool_end", broken)
    record, lines = _logged_run(tmp_path / "b", monkeypatch, replies)

    def summary(r: RunRecord) -> tuple[Any, ...]:
        cost = r.cost
        return (r.status, r.success, cost.tool_calls, cost.tool_errors, cost.agent_llm_calls)

    assert summary(record) == summary(baseline)
    assert any("Error in StepLogger.on_tool_end callback" in line for line in lines)
