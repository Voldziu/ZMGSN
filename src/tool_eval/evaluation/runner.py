"""Runs scenarios against tool levels, one isolated workspace per run, appending JSONL records."""

import hashlib
import json
import re
import secrets
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import openai
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.errors import GraphRecursionError

from tool_eval.agent import build_agent, run_task
from tool_eval.config import Settings, use_settings
from tool_eval.evaluation.checks import CHECKS_VERSION, Lookup, evaluate_review
from tool_eval.evaluation.cost import RunMeter
from tool_eval.evaluation.models import OutcomeScore, RunRecord, RunStatus, Scenario
from tool_eval.evaluation.tracing import (
    start_tracing,
    stop_tracing,
    traced_run,
    tracing_enabled,
)
from tool_eval.exceptions import ExternalServiceError, InvalidInputError
from tool_eval.prompts import load_prompt
from tool_eval.services.reports import REVIEW_PATH
from tool_eval.tools.registry import TOOLSET_VERSION
from tool_eval.utils.logger import get_logger

logger = get_logger(__name__)

_CONTEXT_OVERFLOW = re.compile(r"context[ _](?:size|length|window)", re.IGNORECASE)


def config_fingerprint(settings: Settings) -> str:
    """Hash every setting that changes run results, so incomparable runs can be told apart.

    Args:
        settings: Settings of the run.

    Returns:
        Hex SHA-256 digest.
    """
    payload = {
        "model": settings.llm_model,
        "temperature": settings.agent_temperature,
        "max_steps": settings.agent_max_steps,
        "parallel_tool_calls": settings.effective_parallel_tool_calls,
        "toolset_version": TOOLSET_VERSION,
        "checks_version": CHECKS_VERSION,
        "prompt_version": settings.prompt_version,
        "system_prompt": load_prompt("system", settings.prompt_version).sha256,
        "task_prompt": load_prompt("evaluation", settings.prompt_version).sha256,
        "review_paper_count": settings.review_paper_count,
        "chunk_size": settings.chunk_size,
        "text_max_chars": settings.text_max_chars,
        "pipeline_max_chunks": settings.pipeline_max_chunks,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def arxiv_cache_dir(settings: Settings, runs_dir: Path) -> Path:
    """Directory of the shared arXiv response cache for an evaluation.

    Args:
        settings: Base settings; an explicit `arxiv_cache_dir` wins.
        runs_dir: Directory holding the run workspaces.

    Returns:
        `settings.arxiv_cache_dir`, else `arxiv_cache` next to `runs_dir`.
    """
    return settings.arxiv_cache_dir or runs_dir.parent / "arxiv_cache"


def run_evaluation(
    *,
    scenarios: Sequence[Scenario],
    levels: Sequence[str],
    repeats: int,
    output: Path,
    runs_dir: Path,
    settings: Settings,
    lookup: Lookup,
) -> int:
    """Run every (repeat, scenario, level) combination that has no stored result yet.

    Runs are interleaved by repeat so an interrupted evaluation still has balanced data.
    Runs that ended in `error` (infrastructure failures) are retried on the next invocation;
    `failed` runs are scored and kept.

    Args:
        scenarios: Scenarios to execute.
        levels: Tool level keys to compare.
        repeats: Repetitions per scenario and level.
        output: JSONL file the records are appended to.
        runs_dir: Directory holding one workspace per run (recreated for each run).
        settings: Base settings; the temperature and model come from here.
        lookup: Resolves arXiv ids to metadata for the outcome checks.

    Returns:
        Number of runs executed in this invocation.

    Raises:
        InvalidInputError: If `output` holds finished runs made with a different configuration.
        TrackingServerError: If MLflow tracing is on but its server is not healthy.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _finished_runs(output, config_fingerprint(settings))
    tracing = tracing_enabled()
    if tracing:
        start_tracing()  # fails before any run starts
    try:
        return _execute_runs(
            scenarios=scenarios,
            levels=levels,
            repeats=repeats,
            output=output,
            runs_dir=runs_dir,
            settings=settings,
            lookup=lookup,
            done=done,
        )
    finally:
        if tracing:
            stop_tracing()


def _execute_runs(
    *,
    scenarios: Sequence[Scenario],
    levels: Sequence[str],
    repeats: int,
    output: Path,
    runs_dir: Path,
    settings: Settings,
    lookup: Lookup,
    done: set[str],
) -> int:
    executed = 0
    for repeat in range(1, repeats + 1):
        for scenario in scenarios:
            for level in levels:
                if _run_id(level, scenario.id, repeat) in done:
                    continue
                record = run_single(
                    level=level,
                    scenario=scenario,
                    repeat=repeat,
                    runs_dir=runs_dir,
                    settings=settings,
                    lookup=lookup,
                )
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(record.model_dump_json() + "\n")
                executed += 1
                logger.info(
                    "%s: %s, success=%s, tokens=%d",
                    record.run_id,
                    record.status,
                    record.outcome.success if record.outcome else None,
                    record.cost.total_tokens,
                )
    return executed


def run_single(
    *,
    level: str,
    scenario: Scenario,
    repeat: int,
    runs_dir: Path,
    settings: Settings,
    lookup: Lookup,
) -> RunRecord:
    """Execute one agent run in a fresh workspace and score its output.

    Args:
        level: Tool level key.
        scenario: Scenario to execute.
        repeat: Repetition number (1-based).
        runs_dir: Parent directory for the run's workspace.
        settings: Base settings.
        lookup: Resolves arXiv ids to metadata for the outcome checks.

    Returns:
        The record of this run; infrastructure failures yield `status="error"`, agent-caused
        context overflows `status="failed"`.
    """
    run_id = _run_id(level, scenario.id, repeat)
    workspace = runs_dir / run_id
    shutil.rmtree(workspace, ignore_errors=True)
    run_settings = settings.model_copy(
        update={
            "workspace_dir": workspace,
            "pdf_cache_dir": settings.pdf_cache_dir or runs_dir.parent / "pdf_cache",
            "arxiv_cache_dir": arxiv_cache_dir(settings, runs_dir),
            "arxiv_date_from": scenario.date_from,
            "arxiv_date_to": scenario.date_to,
            "llm_seed": secrets.randbelow(2**31),
        },
    )
    system_prompt = load_prompt("system", run_settings.prompt_version)
    task_prompt = load_prompt("evaluation", run_settings.prompt_version)
    task = task_prompt.render(
        topic=scenario.topic,
        n_papers=str(run_settings.review_paper_count),
        date_from=str(scenario.date_from),
        date_to=str(scenario.date_to),
    )

    started_at = datetime.now(UTC)
    meter = RunMeter()
    status: RunStatus = "completed"
    error: str | None = None
    fingerprint = config_fingerprint(run_settings)
    params = {
        "model": run_settings.llm_model,
        "level": level,
        "scenario": scenario.id,
        "repeat": repeat,
        "seed": run_settings.llm_seed,
        "max_steps": run_settings.agent_max_steps,
        "llm_max_tokens": run_settings.llm_max_tokens,
        "temperature": run_settings.agent_temperature,
        "parallel_tool_calls": run_settings.effective_parallel_tool_calls,
        "prompt_version": run_settings.prompt_version,
        "toolset_version": TOOLSET_VERSION,
        "checks_version": CHECKS_VERSION,
        "config_sha256": fingerprint,
    }
    with traced_run(f"{run_settings.llm_model}/{run_id}", params) as trace:
        with (
            use_settings(run_settings),
            trace.agent_span(
                run_id,
                tags={
                    "model": run_settings.llm_model,
                    "level": level,
                    "scenario": scenario.id,
                    "repeat": str(repeat),
                },
                session_id=f"{run_settings.llm_model}/{scenario.id}",
            ),
        ):
            try:
                agent = build_agent(level, settings=run_settings)
                messages = run_task(
                    agent,
                    task,
                    max_steps=run_settings.agent_max_steps,
                    callbacks=[meter],
                    run_name=run_id,
                    tags=[level, scenario.id],
                    metadata={"level": level, "scenario_id": scenario.id, "repeat": repeat},
                )
                if _hit_token_limit(messages):
                    status = "failed"
                    error = (
                        "final reply hit the completion-token limit "
                        f"(LLM_MAX_TOKENS={run_settings.llm_max_tokens}) without a tool call"
                    )
            except GraphRecursionError:
                status = "step_limit"
            except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                logger.exception("Run %s failed", run_id)
                status = "failed" if _is_context_overflow(exc) else "error"
                error = f"{type(exc).__name__}: {exc}"
            trace.set_status(status, error)

        outcome: OutcomeScore | None = None
        if status != "error":
            try:
                outcome = evaluate_review(
                    workspace / REVIEW_PATH,
                    workspace=workspace,
                    scenario=scenario,
                    expected_papers=run_settings.review_paper_count,
                    lookup=lookup,
                )
            except ExternalServiceError as exc:
                status, error = "error", f"{type(exc).__name__}: {exc}"

        record = RunRecord(
            run_id=run_id,
            level=level,
            scenario_id=scenario.id,
            repeat=repeat,
            status=status,
            error=error,
            outcome=outcome,
            cost=meter.snapshot(),
            model=run_settings.llm_model,
            temperature=run_settings.agent_temperature,
            max_steps=run_settings.agent_max_steps,
            prompt_version=run_settings.prompt_version,
            system_prompt_sha256=system_prompt.sha256,
            task_prompt_sha256=task_prompt.sha256,
            review_paper_count=run_settings.review_paper_count,
            chunk_size=run_settings.chunk_size,
            text_max_chars=run_settings.text_max_chars,
            pipeline_max_chunks=run_settings.pipeline_max_chunks,
            config_sha256=fingerprint,
            seed=run_settings.llm_seed,
            mlflow_run_id=trace.mlflow_run_id,
            trace_id=trace.trace_id,
            date_from=scenario.date_from,
            date_to=scenario.date_to,
            started_at=started_at,
        )
        trace.finish(record, workspace / REVIEW_PATH)
    return record


def _run_id(level: str, scenario_id: str, repeat: int) -> str:
    return f"{level}-{scenario_id}-{repeat:02d}"


def _hit_token_limit(messages: Sequence[AnyMessage] | None) -> bool:
    """Whether the agent's last reply was cut by the token limit instead of ending normally.

    A tool call cut off this way cannot be parsed, so the agent stops with a truncated reply that
    would otherwise look like a finished run.
    """
    if not messages:
        return False
    last = messages[-1]
    return (
        isinstance(last, AIMessage)
        and not last.tool_calls
        and last.response_metadata.get("finish_reason") == "length"
    )


def _is_context_overflow(exc: Exception) -> bool:
    return isinstance(exc, openai.BadRequestError) and bool(_CONTEXT_OVERFLOW.search(str(exc)))


def _finished_runs(output: Path, fingerprint: str) -> set[str]:
    if not output.is_file():
        return set()
    finished: set[str] = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = RunRecord.model_validate_json(line)
            if record.status == "error":
                continue
            if record.config_sha256 != fingerprint:
                msg = (
                    f"{output} holds run {record.run_id} made with a different configuration; "
                    "use another --output file or restore the original settings."
                )
                raise InvalidInputError(msg)
            finished.add(record.run_id)
    return finished
