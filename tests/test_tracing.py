import functools
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.errors import GraphRecursionError

from test_evaluation import SCENARIO, _lookup, _papers, _review_text, _settings, _write_review
from test_metering import _Clock
from tool_eval.evaluation import runner, tracing
from tool_eval.evaluation.cost import RunMeter
from tool_eval.evaluation.report import load_records
from tool_eval.exceptions import TrackingServerError
from tool_eval.main import main
from tool_eval.utils.logger import close_logging, configure_logging


class _FakeSpan:
    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.outputs: dict[str, Any] = {}

    def set_outputs(self, outputs: dict[str, Any]) -> None:
        self.outputs = outputs


class _FakeMlflow(types.ModuleType):
    """Records the MLflow calls the tracing module makes."""

    def __init__(self) -> None:
        super().__init__("mlflow")
        self.calls: list[tuple[str, Any]] = []
        self.runs = 0
        self.spans: list[_FakeSpan] = []
        self.langchain = types.SimpleNamespace(
            autolog=lambda **kwargs: self.calls.append(("autolog", kwargs))
        )

    def set_tracking_uri(self, uri: str) -> None:
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name: str) -> None:
        self.calls.append(("set_experiment", name))

    @contextmanager
    def start_run(self, run_name: str) -> Iterator[Any]:
        self.runs += 1
        self.calls.append(("start_run", run_name))
        yield types.SimpleNamespace(info=types.SimpleNamespace(run_id=f"run-{self.runs}"))

    def log_params(self, params: dict[str, Any]) -> None:
        self.calls.append(("log_params", params))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.calls.append(("log_metrics", metrics))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.calls.append(("set_tags", tags))

    def log_artifact(self, path: str) -> None:
        self.calls.append(("log_artifact", Path(path).name))

    def log_text(self, text: str, artifact_file: str) -> None:
        self.calls.append(("log_text", artifact_file))

    @contextmanager
    def start_span(self, name: str, span_type: str) -> Iterator[_FakeSpan]:
        span = _FakeSpan(f"tr-{self.runs}")
        self.spans.append(span)
        self.calls.append(("start_span", (name, span_type)))
        yield span

    def update_current_trace(self, **kwargs: Any) -> None:
        self.calls.append(("update_current_trace", kwargs))

    def log_feedback(self, **kwargs: Any) -> None:
        self.calls.append(("log_feedback", kwargs))

    def flush_trace_async_logging(self) -> None:
        self.calls.append(("flush", None))

    def named(self, name: str) -> list[Any]:
        return [value for call, value in self.calls if call == name]


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> _FakeMlflow:
    fake = _FakeMlflow()
    entities = types.ModuleType("mlflow.entities")
    entities.SpanType = types.SimpleNamespace(AGENT="AGENT")  # type: ignore[attr-defined]
    entities.AssessmentSource = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    entities.AssessmentSourceType = types.SimpleNamespace(CODE="CODE")  # type: ignore[attr-defined]
    fake.entities = entities  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(tracing, "_server_healthy", lambda uri: True)
    return fake


def _patch_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_eval.config import get_settings

    def run_task(agent: Any, task: str, **kwargs: Any) -> list[Any]:
        _write_review(get_settings().workspace_dir, _review_text())
        return []

    monkeypatch.setattr(runner, "build_agent", lambda level, *, settings=None: object())
    monkeypatch.setattr(runner, "run_task", run_task)


def _evaluate(tmp_path: Path) -> Path:
    output = tmp_path / "results.jsonl"
    runner.run_evaluation(
        scenarios=[SCENARIO],
        levels=["level1", "level3"],
        repeats=1,
        output=output,
        runs_dir=tmp_path / "runs",
        settings=_settings(tmp_path / "unused", llm_model="qwen"),
        lookup=_lookup(_papers()),
    )
    return output


def test_disabled_tracing_never_imports_mlflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setitem(sys.modules, "mlflow", None)  # any import would now fail
    _patch_agent(monkeypatch)
    records = load_records(_evaluate(tmp_path))
    assert not tracing.tracing_enabled()
    assert [(r.mlflow_run_id, r.trace_id) for r in records] == [(None, None), (None, None)]


def test_unhealthy_server_stops_before_any_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    started: list[str] = []
    monkeypatch.setattr(runner, "build_agent", lambda *a, **k: started.append("run"))
    with pytest.raises(TrackingServerError, match=r"http://127\.0\.0\.1:9"):
        _evaluate(tmp_path)
    assert started == []


def test_cli_exits_with_2_when_server_is_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(
        '[{"id": "demo", "topic": "demo topic", "date_from": "2024-01-01",'
        ' "date_to": "2024-06-30"}]',
        encoding="utf-8",
    )
    code = main(["run", "--scenarios", str(scenarios), "--repeats", "1", "--levels", "level1"])
    assert code == 2
    assert "http://127.0.0.1:9" in caplog.text


def test_each_run_is_an_mlflow_run_with_one_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mlflow: _FakeMlflow
) -> None:
    _patch_agent(monkeypatch)
    records = load_records(_evaluate(tmp_path))

    assert fake_mlflow.named("set_tracking_uri") == ["http://localhost:5050"]
    assert fake_mlflow.named("set_experiment") == ["tool-eval"]
    assert fake_mlflow.named("autolog") == [{"silent": True}]
    # Records are in execution order, which is the randomised block order.
    assert sorted(r.run_id for r in records) == ["level1-demo-01", "level3-demo-01"]
    assert fake_mlflow.named("start_run") == [f"qwen/{r.run_id}" for r in records]
    assert fake_mlflow.named("start_span") == [(r.run_id, "AGENT") for r in records]

    params = fake_mlflow.named("log_params")[0]
    assert {"model", "level", "scenario", "repeat", "seed", "max_steps", "config_sha256"} <= set(
        params
    )
    assert params["level"] == records[0].level
    metrics = fake_mlflow.named("log_metrics")[0]
    assert metrics["score"] == 1.0
    assert metrics["success"] == 1.0
    assert {"tool_calls", "tool_errors", "agent_input_tokens", "wall_seconds"} <= set(metrics)
    assert fake_mlflow.named("set_tags")[0]["status"] == "completed"

    trace_updates = fake_mlflow.named("update_current_trace")
    assert trace_updates[0]["tags"]["level"] == records[0].level
    assert trace_updates[0]["session_id"] == "qwen/demo"
    assert fake_mlflow.spans[0].outputs["status"] == "completed"

    feedback = [f for f in fake_mlflow.named("log_feedback") if f["trace_id"] == "tr-1"]
    names = [f["name"] for f in feedback]
    assert names[0] == "score"
    assert "papers_read" in names
    assert len(names) == 1 + len(records[0].outcome.criteria)  # type: ignore[union-attr]
    assert all(f["source"]["source_type"] == "CODE" for f in feedback)

    assert fake_mlflow.named("log_artifact").count("review.md") == 2
    assert fake_mlflow.named("log_text") == ["record.json", "record.json"]
    assert fake_mlflow.named("flush") == [
        None,
        None,
        None,
    ]  # one per run (before assessments), one at the end
    assert [(r.mlflow_run_id, r.trace_id) for r in records] == [
        ("run-1", "tr-1"),
        ("run-2", "tr-2"),
    ]


def test_experiment_name_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mlflow: _FakeMlflow
) -> None:
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "thesis-final")
    _patch_agent(monkeypatch)
    _evaluate(tmp_path)
    assert fake_mlflow.named("set_experiment") == ["thesis-final"]


class _TraceNotStoredYet(Exception):
    """Like MLflow's RESOURCE_DOES_NOT_EXIST for a trace still in the async upload queue."""


def test_assessments_wait_for_the_trace_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mlflow: _FakeMlflow
) -> None:
    flushed = {"done": False}
    real_flush, real_feedback = fake_mlflow.flush_trace_async_logging, fake_mlflow.log_feedback

    def flush() -> None:
        flushed["done"] = True
        real_flush()

    def feedback(**kwargs: Any) -> None:
        if not flushed["done"]:
            raise _TraceNotStoredYet("Trace not found")
        real_feedback(**kwargs)

    monkeypatch.setattr(fake_mlflow, "flush_trace_async_logging", flush)
    monkeypatch.setattr(fake_mlflow, "log_feedback", feedback)
    monkeypatch.setattr(tracing, "_sleep", lambda seconds: None)
    _patch_agent(monkeypatch)
    records = load_records(_evaluate(tmp_path))
    assert len(records) == 2
    assert len(fake_mlflow.named("log_feedback")) == 2 * (1 + len(records[0].outcome.criteria))  # type: ignore[union-attr]


def test_mlflow_failures_never_stop_the_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mlflow: _FakeMlflow
) -> None:
    def feedback(**kwargs: Any) -> None:
        raise _TraceNotStoredYet("Trace not found")

    monkeypatch.setattr(fake_mlflow, "log_feedback", feedback)
    monkeypatch.setattr(tracing, "_sleep", lambda seconds: None)
    _patch_agent(monkeypatch)
    records = load_records(_evaluate(tmp_path))
    assert [r.status for r in records] == ["completed", "completed"]
    assert all(r.trace_id for r in records)


def test_failed_run_creation_falls_back_to_untraced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mlflow: _FakeMlflow
) -> None:
    @contextmanager
    def broken_start_run(run_name: str) -> Iterator[Any]:
        raise ConnectionError("tracking server went away")
        yield  # pragma: no cover

    monkeypatch.setattr(fake_mlflow, "start_run", broken_start_run)
    _patch_agent(monkeypatch)
    records = load_records(_evaluate(tmp_path))
    assert [r.status for r in records] == ["completed", "completed"]
    assert [(r.mlflow_run_id, r.trace_id) for r in records] == [(None, None), (None, None)]


def test_metrics_carry_time_prompt_overhead_and_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mlflow: _FakeMlflow
) -> None:
    from tool_eval.config import get_settings

    def run_task(agent: Any, task: str, *, callbacks: list[Any], **kwargs: Any) -> list[Any]:
        meter = next(c for c in callbacks if isinstance(c, RunMeter))
        call = uuid4()
        meter.on_chat_model_start({}, [[]], run_id=call)
        usage = {"input_tokens": 1200, "output_tokens": 30, "total_tokens": 1230}
        reply = AIMessage("done", usage_metadata=usage)
        meter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=reply)]]), run_id=call)
        _write_review(get_settings().workspace_dir, _review_text())
        return []

    monkeypatch.setattr(runner, "build_agent", lambda level, *, settings=None: object())
    monkeypatch.setattr(runner, "run_task", run_task)
    records = load_records(_evaluate(tmp_path))

    logged = fake_mlflow.named("log_metrics")
    assert len(logged) == 2
    for metrics in logged:
        assert metrics["first_call_input_tokens"] == 1200
        assert metrics["total_tokens"] == 1230
        assert {
            "agent_llm_seconds",
            "tool_llm_seconds",
            "tool_seconds",
            "tool_other_seconds",
            "llm_calls",
            "malformed_tool_calls",
            "wall_seconds",
        } <= set(metrics)
        assert all(isinstance(value, float) for value in metrics.values())
    params = fake_mlflow.named("log_params")
    assert [p["block_position"] for p in params] == [r.block_position for r in records]
    assert {p["order_seed"] for p in params} == {records[0].order_seed}
    assert isinstance(records[0].order_seed, int)


def test_step_limit_run_logs_its_score_but_no_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mlflow: _FakeMlflow
) -> None:
    from tool_eval.config import get_settings

    def run_task(agent: Any, task: str, **kwargs: Any) -> list[Any]:
        _write_review(get_settings().workspace_dir, _review_text())
        raise GraphRecursionError

    monkeypatch.setattr(runner, "build_agent", lambda level, *, settings=None: object())
    monkeypatch.setattr(runner, "run_task", run_task)
    records = load_records(_evaluate(tmp_path))
    assert {r.status for r in records} == {"step_limit"}
    for metrics in fake_mlflow.named("log_metrics"):
        assert (metrics["success"], metrics["score"]) == (0.0, 1.0)


def test_wall_time_excludes_opening_the_mlflow_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mlflow: _FakeMlflow
) -> None:
    from tool_eval.config import get_settings

    clock = _Clock()
    monkeypatch.setattr(runner, "RunMeter", functools.partial(RunMeter, clock=clock))
    real_start_run = fake_mlflow.start_run

    @contextmanager
    def slow_start_run(run_name: str) -> Iterator[Any]:
        clock.advance(10)  # a slow tracking server
        with real_start_run(run_name) as run:
            yield run

    def run_task(agent: Any, task: str, **kwargs: Any) -> list[Any]:
        clock.advance(5)
        _write_review(get_settings().workspace_dir, _review_text())
        return []

    monkeypatch.setattr(fake_mlflow, "start_run", slow_start_run)
    monkeypatch.setattr(runner, "build_agent", lambda level, *, settings=None: object())
    monkeypatch.setattr(runner, "run_task", run_task)
    records = load_records(_evaluate(tmp_path))
    assert [r.cost.wall_seconds for r in records] == [5, 5]


def test_mlflow_warnings_carry_the_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_mlflow: _FakeMlflow
) -> None:
    def broken_metrics(metrics: dict[str, float]) -> None:
        raise ConnectionError("tracking server went away")

    monkeypatch.setattr(fake_mlflow, "log_metrics", broken_metrics)
    _patch_agent(monkeypatch)
    log_file = tmp_path / "eval.log"
    configure_logging(log_file)
    try:
        _evaluate(tmp_path)
    finally:
        close_logging()
    text = log_file.read_text(encoding="utf-8")
    for run_id in ("level1-demo-01", "level3-demo-01"):
        assert f"[{run_id}] MLflow: could not log the results of {run_id}" in text
