import logging
import subprocess
import sys
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

import tool_eval.main as cli
from test_evaluation import (
    GOLDEN_ORDER,
    LEVELS,
    SCENARIO,
    _lookup,
    _papers,
    _patch_agent,
    _record,
    _review_text,
    _settings,
    _ToolCallingFake,
    _write_review,
)
from tool_eval.config import get_settings
from tool_eval.evaluation import runner
from tool_eval.evaluation.cost import RunMeter
from tool_eval.evaluation.report import load_records
from tool_eval.main import main
from tool_eval.utils.logger import close_logging, configure_logging, get_logger, run_context


@pytest.fixture(autouse=True)
def _no_handlers_left_behind() -> Iterator[None]:
    close_logging()
    yield
    close_logging()


@pytest.fixture
def default_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOOL_EVAL_LOG_LEVEL", raising=False)


def _file_handlers(directory: Path) -> list[logging.FileHandler]:
    """File handlers on the root logger that write into `directory` (pytest adds its own)."""
    return [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename).is_relative_to(directory)
    ]


# --- set-up -----------------------------------------------------------------------------


def test_importing_the_package_installs_no_handler() -> None:
    code = (
        "import logging\n"
        "import tool_eval.main, tool_eval.evaluation.runner, tool_eval.services.arxiv\n"
        "from tool_eval.utils.logger import get_logger\n"
        "get_logger('tool_eval.probe').warning('nowhere')\n"
        "print(len(logging.getLogger().handlers), len(logging.getLogger('tool_eval').handlers))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert done.stdout.split() == ["0", "0"]


@pytest.mark.usefixtures("default_level")
def test_file_keeps_debug_lines_and_console_shows_info(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_file = tmp_path / "logs" / "eval.log"
    configure_logging(log_file)
    log = get_logger("tool_eval.probe")
    log.debug("debug detail")
    log.info("info line")
    close_logging()
    console = capsys.readouterr().err
    text = log_file.read_text(encoding="utf-8")
    assert "debug detail" in text
    assert "info line" in text
    assert "info line" in console
    assert "debug detail" not in console


def test_log_level_variable_shows_debug_lines_on_the_console(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOOL_EVAL_LOG_LEVEL", "debug")
    configure_logging(tmp_path / "eval.log")
    get_logger("tool_eval.probe").debug("debug detail")
    close_logging()
    assert "debug detail" in capsys.readouterr().err


@pytest.mark.usefixtures("default_level")
def test_a_second_set_up_closes_the_first_file(tmp_path: Path) -> None:
    first, second = tmp_path / "a.log", tmp_path / "b.log"
    configure_logging(first)
    [first_handler] = _file_handlers(tmp_path)
    configure_logging(second)
    get_logger("tool_eval.probe").info("only in b")
    [second_handler] = _file_handlers(tmp_path)
    close_logging()
    assert first_handler.stream is None  # closed
    assert second_handler.baseFilename == str(second)
    assert "only in b" not in first.read_text(encoding="utf-8")
    assert "only in b" in second.read_text(encoding="utf-8")
    assert _file_handlers(tmp_path) == []


@pytest.mark.usefixtures("default_level")
def test_library_warnings_reach_the_file(tmp_path: Path) -> None:
    log_file = tmp_path / "eval.log"
    configure_logging(log_file)
    library = logging.getLogger("langchain_core.callbacks.manager")
    library.warning("Error in StepLogger.on_tool_end callback")
    library.info("library chatter")
    close_logging()
    text = log_file.read_text(encoding="utf-8")
    assert "Error in StepLogger.on_tool_end callback" in text
    assert "library chatter" not in text


# --- command line ---------------------------------------------------------------------------


def _scenarios(tmp_path: Path) -> Path:
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(
        '[{"id": "demo", "topic": "demo topic", "date_from": "2024-01-01",'
        ' "date_to": "2024-06-30"}]',
        encoding="utf-8",
    )
    return scenarios


def _fake_evaluation(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the evaluation with one that logs a line per invocation, like the runner does."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    calls: list[dict[str, Any]] = []

    def run_evaluation(**kwargs: Any) -> int:
        calls.append(kwargs)
        get_logger("tool_eval.evaluation.runner").debug("invocation %d", len(calls))
        return 0

    monkeypatch.setattr(cli, "run_evaluation", run_evaluation)
    return calls


def test_run_logs_next_to_its_results_and_appends_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_evaluation(monkeypatch)
    output = tmp_path / "results" / "presets-x.jsonl"
    args = ["run", "--scenarios", str(_scenarios(tmp_path)), "--output", str(output)]
    assert main(args) == 0
    assert main(args) == 0
    text = (tmp_path / "results" / "presets-x.log").read_text(encoding="utf-8")
    assert text.index("invocation 1") < text.index("invocation 2")
    assert f"log in {output.with_suffix('.log')}" in text
    assert _file_handlers(tmp_path) == []  # closed when the command returned


def test_report_writes_no_log_file(tmp_path: Path) -> None:
    results = tmp_path / "r.jsonl"
    results.write_text(
        _record("level1", "a", 1, score=1.0, tokens=100).model_dump_json() + "\n",
        encoding="utf-8",
    )
    assert main(["report", str(results)]) == 0
    assert list(tmp_path.rglob("*.log")) == []


def test_run_rejects_an_output_that_ends_in_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _fake_evaluation(monkeypatch)
    output = tmp_path / "x.log"
    args = ["run", "--scenarios", str(_scenarios(tmp_path)), "--output", str(output)]
    assert main(args) == 2
    assert calls == []
    assert not output.exists()
    assert "must not end in .log" in capsys.readouterr().err


# --- run ID -------------------------------------------------------------------------------


@pytest.mark.usefixtures("default_level")
def test_lines_written_inside_a_run_carry_its_id(tmp_path: Path) -> None:
    log_file = tmp_path / "eval.log"
    configure_logging(log_file)
    log = get_logger("tool_eval.probe")
    log.info("before")
    with run_context("level1-demo-01"):
        log.info("inside")
    log.info("after")
    close_logging()
    before, inside, after = log_file.read_text(encoding="utf-8").splitlines()
    assert inside.endswith(" [level1-demo-01] inside")
    assert before.endswith(" before")
    assert after.endswith(" after")
    assert "[" not in before + after


@pytest.mark.usefixtures("default_level")
def test_the_run_id_reaches_lines_that_tools_log_from_worker_threads(tmp_path: Path) -> None:
    arxiv_log = logging.getLogger("tool_eval.services.arxiv")
    threads: set[str] = set()

    @tool
    def flaky(attempt: int) -> str:
        """Log a retry warning the way the arXiv client does."""
        threads.add(threading.current_thread().name)
        arxiv_log.warning("arXiv request failed (HTTP 429); attempt %d", attempt)
        return "ok"

    calls = [{"name": "flaky", "args": {"attempt": i}, "id": f"call-{i}"} for i in (1, 2)]
    model = _ToolCallingFake(messages=iter([AIMessage("", tool_calls=calls), AIMessage("done")]))
    log_file = tmp_path / "eval.log"
    configure_logging(log_file)
    with run_context("level3-demo-01"):
        create_agent(model=model, tools=[flaky]).invoke({"messages": [HumanMessage("go")]})
    close_logging()
    lines = [
        line
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if "arXiv request failed" in line
    ]
    assert len(lines) == 2
    assert all("[level3-demo-01] arXiv request failed" in line for line in lines)
    assert threads - {threading.main_thread().name}  # the tools ran in worker threads


# --- runner lines ---------------------------------------------------------------------------


@pytest.fixture
def log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("TOOL_EVAL_LOG_LEVEL", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    path = tmp_path / "eval.log"
    configure_logging(path)
    return path


def _lines(log_file: Path) -> list[str]:
    close_logging()
    return log_file.read_text(encoding="utf-8").splitlines()


def _evaluate(tmp_path: Path, *, levels: Sequence[str] = ("level1",), repeats: int = 1) -> Path:
    output = tmp_path / "results.jsonl"
    runner.run_evaluation(
        scenarios=[SCENARIO],
        levels=levels,
        repeats=repeats,
        output=output,
        runs_dir=tmp_path / "runs",
        settings=_settings(tmp_path / "unused", llm_model="qwen"),
        lookup=_lookup(_papers()),
        order_seed=7,
    )
    return output


def _write_reviews(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_agent(monkeypatch, lambda: _write_review(get_settings().workspace_dir, _review_text()))


def test_a_failing_run_logs_its_start_and_its_failure_with_its_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log_file: Path
) -> None:
    def server_down() -> None:
        raise ConnectionError("vLLM server unreachable")

    _patch_agent(monkeypatch, server_down)
    _evaluate(tmp_path)
    lines = _lines(log_file)
    own = [line for line in lines if "[level1-demo-01] " in line]
    assert "start: position" in own[0]
    assert any("[level1-demo-01] Run level1-demo-01 failed" in line for line in own)
    assert "end: error" in own[-1]


def test_the_evaluation_start_line_counts_finished_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log_file: Path
) -> None:
    _write_reviews(monkeypatch)
    _evaluate(tmp_path, levels=["level1", "level2"], repeats=2)
    _evaluate(tmp_path, levels=LEVELS, repeats=2)
    starts = [line for line in _lines(log_file) if " evaluation: " in line]
    assert starts[0].endswith(
        "evaluation: model=qwen, levels=level1 level2, scenarios=demo, repeats=2, "
        "order_seed=7, finished=0, to_run=4"
    )
    assert starts[1].endswith(
        "evaluation: model=qwen, levels=level1 level2 level3, scenarios=demo, repeats=2, "
        "order_seed=7, finished=4, to_run=2"
    )


def test_each_run_starts_with_its_position_and_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log_file: Path
) -> None:
    _write_reviews(monkeypatch)
    records = load_records(_evaluate(tmp_path, levels=LEVELS))
    lines = _lines(log_file)
    for record in records:
        own = [line for line in lines if f"[{record.run_id}] " in line]
        assert own[0].endswith(f"start: position {record.block_position}/3, seed={record.seed}")
    second = f"{GOLDEN_ORDER[1]}-demo-01"
    assert any(f"[{second}] start: position 2/3, seed=" in line for line in lines)


def test_the_end_line_repeats_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log_file: Path
) -> None:
    def run_task(agent: Any, task: str, *, callbacks: list[Any], **kwargs: Any) -> list[Any]:
        meter = next(c for c in callbacks if isinstance(c, RunMeter))
        tool_run = uuid4()
        meter.on_tool_start({"name": "search_arxiv"}, "", run_id=tool_run)
        meter.on_tool_error(ValueError("no papers"), run_id=tool_run)
        raise GraphRecursionError

    monkeypatch.setattr(runner, "build_agent", lambda level, *, settings=None: object())
    monkeypatch.setattr(runner, "run_task", run_task)
    [record] = load_records(_evaluate(tmp_path))
    cost = record.cost
    assert (record.status, record.success, cost.tool_calls, cost.tool_errors) == (
        "step_limit",
        False,
        1,
        1,
    )
    expected = (
        f"[{record.run_id}] end: step_limit, success=False, tokens={cost.total_tokens}, "
        f"wall={cost.wall_seconds:.1f} s, tool_calls=1, tool_errors=1"
    )
    assert any(line.endswith(expected) for line in _lines(log_file))
