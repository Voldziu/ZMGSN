import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import tool

from test_evaluation import (
    SCENARIO,
    _lookup,
    _papers,
    _record,
    _review_text,
    _settings,
    _ToolCallingFake,
    _write_review,
)
from tool_eval.agent import _report_domain_error
from tool_eval.config import get_settings
from tool_eval.evaluation import runner
from tool_eval.evaluation.loop_guard import LoopGuard
from tool_eval.evaluation.models import RunRecord
from tool_eval.evaluation.report import load_records, render_markdown, summarize
from tool_eval.exceptions import AgentLoopError, ExternalServiceError, ToolEvalError
from tool_eval.utils.logger import close_logging, configure_logging

# --- unit tests: the callbacks driven directly -------------------------------------------


def _guard(repeated: int = 5, errors: int = 10) -> LoopGuard:
    return LoopGuard(max_repeated_calls=repeated, max_tool_error_streak=errors)


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": str(uuid4())}


def _reply(guard: LoopGuard, *calls: dict[str, Any]) -> None:
    """One reply of the agent's model, requesting `calls`."""
    run = uuid4()
    guard.on_chat_model_start({}, [[]], run_id=run)
    message = AIMessage("", tool_calls=list(calls))
    guard.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run)


def _tool(guard: LoopGuard, name: str, *, fail: bool = False) -> UUID:
    run = uuid4()
    guard.on_tool_start({"name": name}, "", run_id=run)
    if fail:
        guard.on_tool_error(RuntimeError("boom"), run_id=run)
    else:
        guard.on_tool_end("ok", run_id=run)
    return run


def test_loop_error_is_not_reported_back_to_the_agent() -> None:
    assert not issubclass(AgentLoopError, ToolEvalError)


def test_fifth_identical_call_stops_the_run() -> None:
    guard = _guard()
    for _ in range(4):
        _reply(guard, _call("conduct_literature_review", query="rag"))
    with pytest.raises(AgentLoopError, match=r"5 times in a row: conduct_literature_review\("):
        _reply(guard, _call("conduct_literature_review", query="rag"))


def test_different_arguments_or_a_call_in_between_reset_the_count() -> None:
    guard = _guard()
    for i in range(10):
        _reply(guard, _call("search_arxiv", query=f"q{i}"))
    for _ in range(4):
        _reply(guard, _call("read_file", path="a.md"))
    _reply(guard, _call("search_arxiv", query="x"))
    for _ in range(4):
        _reply(guard, _call("read_file", path="a.md"))


def test_identical_calls_in_one_reply_count() -> None:
    with pytest.raises(AgentLoopError):
        _reply(_guard(), *[_call("read_file", path="a.md") for _ in range(5)])


def test_error_streak_stops_before_the_next_reply() -> None:
    guard = _guard()
    for _ in range(10):
        _tool(guard, "conduct_literature_review", fail=True)
    with pytest.raises(AgentLoopError, match="10 tool calls in a row failed.*conduct_literature"):
        guard.on_chat_model_start({}, [[]], run_id=uuid4())


def test_a_success_resets_the_error_streak() -> None:
    guard = _guard()
    for _ in range(9):
        _tool(guard, "search_arxiv", fail=True)
    _tool(guard, "search_arxiv")
    for _ in range(9):
        _tool(guard, "search_arxiv", fail=True)
    _reply(guard)


def test_model_calls_inside_a_tool_count_for_neither_check() -> None:
    guard = _guard(repeated=2, errors=1)
    tool_run = uuid4()
    guard.on_tool_start({"name": "conduct_literature_review"}, "", run_id=tool_run)
    _tool(guard, "nested", fail=True)  # a failing nested tool; the outer one is still active
    for _ in range(3):
        _reply(guard, _call("summarise", text="same"))
    guard.on_tool_end("ok", run_id=tool_run)
    _reply(guard, _call("read_file", path="a.md"))


def test_zero_switches_a_check_off() -> None:
    guard = _guard(repeated=0, errors=0)
    for _ in range(20):
        _reply(guard, _call("read_file", path="a.md"))
        _tool(guard, "read_file", fail=True)
    _reply(guard)


# --- end to end: the real create_agent loop through the runner ---------------------------


@tool
def read_file(path: str) -> str:
    """Read a file from the workspace."""
    return f"content of {path}"


@tool
def search_arxiv(query: str) -> str:
    """Search arXiv."""
    raise ExternalServiceError(f"no papers for {query}")


@tool
def write_review() -> str:
    """Write the review."""
    _write_review(get_settings().workspace_dir, _review_text())
    return "written"


class _Counted:
    """Scripted replies that count how often the agent's model was called."""

    def __init__(self, replies: Callable[[int], AIMessage]) -> None:
        self.calls = 0
        self._replies = replies

    def __iter__(self) -> Iterator[AIMessage]:
        while True:
            self.calls += 1
            yield self._replies(self.calls)


def _evaluate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: _Counted,
    **settings: Any,
) -> tuple[RunRecord, list[str]]:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    def build_agent(level: str, *, settings: Any = None) -> Any:
        return create_agent(
            model=_ToolCallingFake(messages=iter(script)),
            tools=[read_file, search_arxiv, write_review],
            middleware=[ToolErrorMiddleware(_report_domain_error)],
        )

    monkeypatch.setattr(runner, "build_agent", build_agent)
    output, log_file = tmp_path / "results.jsonl", tmp_path / "results.log"
    configure_logging(log_file)
    try:
        runner.run_evaluation(
            scenarios=[SCENARIO],
            levels=["level1"],
            repeats=1,
            output=output,
            runs_dir=tmp_path / "runs",
            settings=_settings(tmp_path / "unused", llm_model="qwen", **settings),
            lookup=_lookup(_papers()),
            order_seed=7,
        )
    finally:
        close_logging()
    [record] = load_records(output)
    return record, log_file.read_text(encoding="utf-8").splitlines()


def _calls(*calls: dict[str, Any]) -> AIMessage:
    return AIMessage("", tool_calls=list(calls))


def test_repeating_agent_ends_as_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _Counted(lambda _: _calls(_call("read_file", path="notes.md")))
    record, lines = _evaluate(tmp_path, monkeypatch, script)
    assert record.status == "loop"
    assert record.error is not None and "read_file" in record.error
    assert not record.success
    assert record.cost.tool_calls == 4
    assert record.max_repeated_calls == 5
    assert record.max_tool_error_streak == 10
    assert any("WARNING" in line and "loop guard stopped the run" in line for line in lines)
    assert any("end: loop, success=False, " in line for line in lines)


def test_failing_agent_ends_as_loop_before_its_next_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _Counted(lambda i: _calls(_call("search_arxiv", query=f"q{i}")))
    record, _ = _evaluate(tmp_path, monkeypatch, script)
    assert record.status == "loop"
    assert record.cost.tool_errors == 10
    assert script.calls == 10


def test_finishing_agent_is_unaffected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replies = [_calls(_call("read_file", path="a.md")), AIMessage("done")]
    script = _Counted(lambda i: replies[i - 1])
    record, _ = _evaluate(tmp_path, monkeypatch, script)
    assert record.status == "completed"


def test_alternating_calls_still_end_at_the_step_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _Counted(lambda i: _calls(_call("read_file", path=f"{i % 2}.md")))
    record, _ = _evaluate(tmp_path, monkeypatch, script, agent_max_steps=6)
    assert record.status == "step_limit"


def test_loop_after_a_complete_review_keeps_its_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _Counted(lambda _: _calls(_call("write_review")))
    record, _ = _evaluate(tmp_path, monkeypatch, script)
    assert record.status == "loop"
    assert record.outcome is not None and record.outcome.score == 1.0
    assert not record.success


def test_loop_warning_reaches_the_console(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    script = _Counted(lambda _: _calls(_call("read_file", path="notes.md")))
    _evaluate(tmp_path, monkeypatch, script)
    assert any("loop guard stopped the run" in r.getMessage() for r in caplog.records)


# --- report -------------------------------------------------------------------------------


def test_quality_table_counts_loop_and_step_limit_runs() -> None:
    records = [
        *(_record("level3", "a", i, score=0.5, tokens=10, status="loop") for i in range(1, 8)),
        _record("level3", "a", 8, score=0.5, tokens=10, status="step_limit"),
        _record("level3", "a", 9, score=1.0, tokens=10),
    ]
    [summary] = summarize(records)
    assert (summary.loop_runs, summary.step_limit_runs) == (7, 1)
    assert "| loop |" in render_markdown(records)


def test_older_results_have_no_loop_runs() -> None:
    [summary] = summarize([_record("level1", "a", 1, score=1.0, tokens=10)])
    assert summary.loop_runs == 0
