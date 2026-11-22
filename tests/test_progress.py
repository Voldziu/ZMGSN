import io
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from test_evaluation import SCENARIO, _lookup, _papers, _settings
from test_steps import _build_agent
from tool_eval.evaluation import runner
from tool_eval.evaluation.models import Scenario
from tool_eval.evaluation.progress import PRESET_POSITION_VARIABLE, Progress, preset_position
from tool_eval.utils.logger import close_logging, configure_logging, get_logger


def _progress(stream: io.StringIO | None, **extra: int) -> Progress:
    return Progress(
        model="qwen",
        planned_runs=extra.get("planned", 6),
        finished_runs=extra.get("finished", 0),
        repeats=2,
        scenarios=3,
        file=stream,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2/5", (2, 5)), (" 3 / 3 ", (3, 3)), ("", (1, 1)), ("6/5", (1, 1)), ("x", (1, 1))],
)
def test_preset_position(monkeypatch: pytest.MonkeyPatch, value: str, expected: object) -> None:
    monkeypatch.setenv(PRESET_POSITION_VARIABLE, value)
    assert preset_position() == expected


def test_bars_show_model_repeat_scenario_and_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PRESET_POSITION_VARIABLE, "2/5")
    stream = io.StringIO()
    with _progress(stream) as progress:
        progress.repeat(2)
        progress.scenario(3, "rlhf-2024")
        progress.run("level2", 1, 3)
        progress.run_done()
    text = stream.getvalue()
    for part in ("model    2/5 qwen", "runs     qwen", "repeat   2/2", "rlhf-2024", "level2 (1/3)"):
        assert part in text


def test_model_bar_without_preset_position(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PRESET_POSITION_VARIABLE, raising=False)
    stream = io.StringIO()
    with _progress(stream):
        pass
    assert "model    1/1 qwen" in stream.getvalue()


def test_disabled_progress_draws_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    with _progress(None) as progress:
        progress.repeat(1)
        progress.scenario(1, "a")
        progress.run("level1", 1, 3)
        progress.run_done()
    assert not progress.enabled
    assert capsys.readouterr().err == ""


def _evaluate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: io.StringIO | None) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(runner, "build_agent", _build_agent([AIMessage("done")] * 10))
    second = Scenario(
        id="demo-2", topic="second topic", date_from=SCENARIO.date_from, date_to=SCENARIO.date_to
    )
    runner.run_evaluation(
        scenarios=[SCENARIO, second],
        levels=["level1", "level2"],
        repeats=1,
        output=tmp_path / "results.jsonl",
        runs_dir=tmp_path / "runs",
        settings=_settings(tmp_path / "unused", llm_model="qwen"),
        lookup=_lookup(_papers()),
        order_seed=7,
        progress_file=stream,
    )


def test_resumed_evaluation_starts_the_runs_bar_at_the_finished_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _evaluate(tmp_path, monkeypatch, None)
    output = tmp_path / "results.jsonl"
    lines = output.read_text(encoding="utf-8").splitlines()
    output.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")  # 3 of 4 finished
    stream = io.StringIO()
    _evaluate(tmp_path, monkeypatch, stream)
    text = stream.getvalue()
    assert "3/4" in text
    assert "4/4" in text
    assert len([json.loads(line) for line in output.read_text().splitlines()]) == 4


def test_no_bars_without_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _evaluate(tmp_path, monkeypatch, None)
    assert "runs     qwen" not in capsys.readouterr().err


def test_console_lines_keep_their_run_id_through_tqdm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tool_eval.utils.logger import run_context

    configure_logging(None)
    try:
        with run_context("r1"):
            get_logger("tool_eval.test").info("hello")
    finally:
        close_logging()
    assert "[r1] hello" in capsys.readouterr().err
