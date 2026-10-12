"""Runs scenarios against tool levels, one isolated workspace per run, appending JSONL records."""

import hashlib
import json
import logging
import re
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import openai
from langgraph.errors import GraphRecursionError

from tool_eval.agent import build_agent, run_task
from tool_eval.config import Settings, use_settings
from tool_eval.evaluation.checks import Lookup, evaluate_review
from tool_eval.evaluation.cost import RunMeter
from tool_eval.evaluation.models import OutcomeScore, RunRecord, RunStatus, Scenario
from tool_eval.exceptions import ExternalServiceError, InvalidInputError
from tool_eval.prompts import load_prompt
from tool_eval.services.reports import REVIEW_PATH

logger = logging.getLogger(__name__)

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
        "prompt_version": settings.prompt_version,
        "system_prompt": load_prompt("system", settings.prompt_version).sha256,
        "task_prompt": load_prompt("evaluation", settings.prompt_version).sha256,
        "review_paper_count": settings.review_paper_count,
        "chunk_size": settings.chunk_size,
        "text_max_chars": settings.text_max_chars,
        "pipeline_max_chunks": settings.pipeline_max_chunks,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _finished_runs(output, config_fingerprint(settings))
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
            "arxiv_date_from": scenario.date_from,
            "arxiv_date_to": scenario.date_to,
        }
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
    with use_settings(run_settings):
        try:
            agent = build_agent(level, settings=run_settings)
            run_task(agent, task, max_steps=run_settings.agent_max_steps, callbacks=[meter])
        except GraphRecursionError:
            status = "step_limit"
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            logger.exception("Run %s failed", run_id)
            status = "failed" if _is_context_overflow(exc) else "error"
            error = f"{type(exc).__name__}: {exc}"

    outcome: OutcomeScore | None = None
    if status != "error":
        try:
            outcome = evaluate_review(
                workspace / REVIEW_PATH,
                scenario=scenario,
                expected_papers=run_settings.review_paper_count,
                lookup=lookup,
            )
        except ExternalServiceError as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"

    return RunRecord(
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
        config_sha256=config_fingerprint(run_settings),
        date_from=scenario.date_from,
        date_to=scenario.date_to,
        started_at=started_at,
    )


def _run_id(level: str, scenario_id: str, repeat: int) -> str:
    return f"{level}-{scenario_id}-{repeat:02d}"


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
