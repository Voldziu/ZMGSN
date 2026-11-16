"""Optional MLflow tracking of evaluation runs, switched on by `MLFLOW_TRACKING_URI`.

Each evaluation run becomes one MLflow run (params, metrics, tags, the review as an artifact)
with exactly one trace: a root `AGENT` span named after the run, under which
`mlflow.langchain.autolog` nests every agent model call, tool call and model call made inside a
tool. The outcome score and every criterion are attached to the trace as code-sourced
assessments. `mlflow` is imported lazily, so the disabled path never needs it.

Tracing is best effort: once the server has passed the start-up health check, an MLflow
failure is logged as a warning and never stops the evaluation or changes its results.
"""

import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from tool_eval.evaluation.models import RunRecord
from tool_eval.exceptions import TrackingServerError
from tool_eval.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_EXPERIMENT = "tool-eval"
_CHECKS_SOURCE = "tool_eval.evaluation.checks"
_MAX_TAG_CHARS = 5000  # MLflow tag values are limited in length
_FEEDBACK_ATTEMPTS = 3
_FEEDBACK_RETRY_SECONDS = 2.0
_sleep = time.sleep  # patched in tests


def tracing_enabled() -> bool:
    """Whether evaluation runs are recorded in MLflow.

    Returns:
        True when `MLFLOW_TRACKING_URI` is set to a non-empty value.
    """
    return bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip())


def start_tracing() -> None:
    """Check the tracking server, select the experiment and enable LangChain autologging.

    The experiment comes from `MLFLOW_EXPERIMENT_NAME` (default `tool-eval`).

    Raises:
        TrackingServerError: If the server at `MLFLOW_TRACKING_URI` is not healthy.
    """
    uri = os.environ["MLFLOW_TRACKING_URI"].strip()
    if not _server_healthy(uri):
        msg = (
            f"MLflow tracking server at {uri} is not healthy; start it with `make mlflow-up` "
            "or unset MLFLOW_TRACKING_URI to run without tracing."
        )
        raise TrackingServerError(msg)
    import mlflow  # noqa: PLC0415 - optional dependency, only needed when tracing is on

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT)
    mlflow.langchain.autolog(silent=True)


def stop_tracing() -> None:
    """Send traces still queued for asynchronous upload (MLflow logs traces asynchronously)."""
    import mlflow  # noqa: PLC0415

    mlflow.flush_trace_async_logging()


class RunTrace:
    """Handle on one evaluation run's MLflow run and trace; every method is a no-op when off."""

    def __init__(self, mlflow: Any | None, mlflow_run_id: str | None) -> None:
        """Wrap an active MLflow run, or nothing when tracing is off.

        Args:
            mlflow: The `mlflow` module, or `None` when tracing is off.
            mlflow_run_id: ID of the active MLflow run.
        """
        self._mlflow = mlflow
        self._span: Any = None
        self.mlflow_run_id = mlflow_run_id
        self.trace_id: str | None = None

    @contextmanager
    def agent_span(self, name: str, *, tags: Mapping[str, str], session_id: str) -> Iterator[None]:
        """Open the root span that autologged agent, tool and model spans nest under.

        Args:
            name: Span name, the evaluation run ID.
            tags: Trace tags for filtering (model, level, scenario, repeat).
            session_id: Trace session, `<model>/<scenario>`, grouping repeats.

        Yields:
            Nothing; the trace ID is available afterwards as `trace_id`.
        """
        if self._mlflow is None:
            yield
            return
        from mlflow.entities import SpanType  # noqa: PLC0415

        with ExitStack() as stack:
            try:
                span = stack.enter_context(
                    self._mlflow.start_span(name=name, span_type=SpanType.AGENT)
                )
                self._span = span
                self.trace_id = span.trace_id
                self._mlflow.update_current_trace(tags=dict(tags), session_id=session_id)
            except Exception:  # noqa: BLE001 - tracing must never stop an evaluation
                logger.warning("MLflow: could not open the trace for %s", name, exc_info=True)
            try:
                yield
            finally:
                self._span = None

    def set_status(self, status: str, error: str | None) -> None:
        """Record the run's status on the trace (inside `agent_span`).

        Args:
            status: Run status, for example `completed` or `step_limit`.
            error: Error message, if any.
        """
        if self._mlflow is None or self._span is None:
            return
        try:
            self._mlflow.update_current_trace(tags={"status": status})
            self._span.set_outputs({"status": status, "error": error})
        except Exception:  # noqa: BLE001 - tracing must never stop an evaluation
            logger.warning("MLflow: could not record the run status", exc_info=True)

    def finish(self, record: RunRecord, review_file: Path) -> None:
        """Log the run's metrics, tags, artifacts and per-criterion assessments.

        Args:
            record: The finished run record, including its MLflow IDs.
            review_file: The review the agent wrote; logged as an artifact if it exists.
        """
        if self._mlflow is None:
            return
        try:
            self._log_run(record, review_file)
            self._log_assessments(record)
        except Exception:  # noqa: BLE001 - tracing must never stop an evaluation
            logger.warning("MLflow: could not log the results of %s", record.run_id, exc_info=True)

    def _log_run(self, record: RunRecord, review_file: Path) -> None:
        mlflow = self._mlflow
        assert mlflow is not None
        outcome = record.outcome
        mlflow.log_metrics(
            {
                **{key: float(value) for key, value in record.cost.model_dump().items()},
                "score": outcome.score if outcome else 0.0,
                "success": 1.0 if outcome and outcome.success else 0.0,
            }
        )
        mlflow.set_tags({"status": record.status, "error": (record.error or "")[:_MAX_TAG_CHARS]})
        if review_file.is_file():
            mlflow.log_artifact(str(review_file))
        mlflow.log_text(record.model_dump_json(indent=2), "record.json")

    def _log_assessments(self, record: RunRecord) -> None:
        mlflow = self._mlflow
        outcome = record.outcome
        if mlflow is None or outcome is None or self.trace_id is None:
            return
        from mlflow.entities import AssessmentSource, AssessmentSourceType  # noqa: PLC0415

        # Traces upload asynchronously, and the server rejects assessments for a trace it has
        # not stored yet ("Trace ... not found"), so drain the upload queue first.
        mlflow.flush_trace_async_logging()
        source = AssessmentSource(source_type=AssessmentSourceType.CODE, source_id=_CHECKS_SOURCE)
        log = partial(mlflow.log_feedback, trace_id=self.trace_id, source=source)
        _retry(partial(log, name="score", value=outcome.score))
        for criterion in outcome.criteria:
            _retry(
                partial(
                    log,
                    name=criterion.name,
                    value=criterion.passed,
                    rationale=criterion.detail or None,
                )
            )


@contextmanager
def traced_run(run_name: str, params: Mapping[str, Any]) -> Iterator[RunTrace]:
    """Open one MLflow run for an evaluation run (a no-op handle when tracing is off).

    Args:
        run_name: MLflow run name, `<model>/<run_id>`.
        params: Parameters that describe the run's configuration.

    Yields:
        The run's trace handle.
    """
    if not tracing_enabled():
        yield RunTrace(None, None)
        return
    import mlflow  # noqa: PLC0415

    with ExitStack() as stack:
        try:
            run = stack.enter_context(mlflow.start_run(run_name=run_name))
            mlflow.log_params(dict(params))
            trace = RunTrace(mlflow, run.info.run_id)
        except Exception:  # noqa: BLE001 - tracing must never stop an evaluation
            logger.warning(
                "MLflow: could not start run %s; continuing untraced", run_name, exc_info=True
            )
            trace = RunTrace(None, None)
        yield trace


def _retry(call: Callable[[], object]) -> None:
    for attempt in range(1, _FEEDBACK_ATTEMPTS + 1):
        try:
            call()
        except Exception:
            if attempt == _FEEDBACK_ATTEMPTS:
                raise
            _sleep(_FEEDBACK_RETRY_SECONDS)
        else:
            return


def _server_healthy(uri: str) -> bool:
    try:
        response = httpx.get(f"{uri.rstrip('/')}/health", timeout=5)
    except httpx.HTTPError:
        return False
    return response.status_code == httpx.codes.OK and response.text.strip() == "OK"
