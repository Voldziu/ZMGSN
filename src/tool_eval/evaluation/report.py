"""Aggregation of run records into per-level metrics and a Markdown report."""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, stdev

from pydantic import BaseModel, ConfigDict

from tool_eval.evaluation.models import RunRecord
from tool_eval.services.formatting import format_markdown_table


class LevelSummary(BaseModel):
    """Metrics of one tool level over all valid (non-`error`) runs."""

    model_config = ConfigDict(frozen=True)

    level: str
    runs: int
    error_runs: int
    step_limit_runs: int
    success_rate: float
    mean_score: float
    mean_total_tokens: float
    mean_tool_internal_tokens: float
    mean_llm_calls: float
    mean_tool_calls: float
    mean_wall_seconds: float
    # Repeatability: spread across repeats of the same scenario, averaged over scenarios.
    repeatability_score_std: float
    repeatability_tokens_cv: float
    # Consistency: spread of per-scenario means across scenarios.
    consistency_score_std: float
    consistency_tokens_cv: float


def load_records(path: Path) -> list[RunRecord]:
    """Read all records from a JSONL results file.

    Args:
        path: JSONL file written by the runner.

    Returns:
        Records in file order.
    """
    return [
        RunRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(records: Sequence[RunRecord]) -> list[LevelSummary]:
    """Compute one summary per level, in order of first appearance.

    Args:
        records: Run records, possibly from several levels.

    Returns:
        Per-level summaries; levels with no valid run have all metrics at zero.
    """
    by_level: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_level[record.level].append(record)
    return [_summarize_level(level, items) for level, items in by_level.items()]


def render_markdown(records: Sequence[RunRecord]) -> str:
    """Render the full report: level comparison, per-scenario table and failed runs.

    Args:
        records: Run records to report on.

    Returns:
        Markdown text.
    """
    summaries = summarize(records)
    metric_rows = [
        [
            s.level,
            str(s.runs),
            f"{s.success_rate:.0%}",
            f"{s.mean_score:.2f}",
            f"{s.mean_total_tokens:.0f}",
            f"{s.mean_tool_internal_tokens:.0f}",
            f"{s.mean_llm_calls:.1f}",
            f"{s.mean_tool_calls:.1f}",
            f"{s.mean_wall_seconds:.0f}",
            str(s.step_limit_runs),
        ]
        for s in summaries
    ]
    stability_rows = [
        [
            s.level,
            f"{s.repeatability_score_std:.3f}",
            f"{s.repeatability_tokens_cv:.2f}",
            f"{s.consistency_score_std:.3f}",
            f"{s.consistency_tokens_cv:.2f}",
        ]
        for s in summaries
    ]
    levels = [s.level for s in summaries]
    sections = [
        "# Evaluation report",
        "## Per level",
        format_markdown_table(
            [
                "level",
                "runs",
                "success",
                "score",
                "tokens",
                "tool-internal tokens",
                "LLM calls",
                "tool calls",
                "wall s*",
                "step limit",
            ],
            metric_rows,
        ),
        "*Wall time depends on network and arXiv throttling; treat it as indicative only.",
        "## Repeatability and consistency",
        "Repeatability = mean over scenarios of the spread across repeats. "
        "Consistency = spread of per-scenario means across scenarios. "
        "`std` is of the score, `cv` is the coefficient of variation of total tokens. "
        "Lower is more stable.",
        format_markdown_table(
            [
                "level",
                "repeat. score std",
                "repeat. tokens cv",
                "consist. score std",
                "consist. tokens cv",
            ],
            stability_rows,
        ),
        "## Per scenario (success rate / mean tokens)",
        format_markdown_table(["scenario", *levels], _scenario_rows(records, levels)),
    ]
    failed = [r for r in records if r.status == "error"]
    if failed:
        lines = "\n".join(f"- `{r.run_id}`: {r.error}" for r in failed)
        sections += [
            "## Runs with infrastructure errors (excluded from all metrics)",
            lines,
        ]
    return "\n\n".join(sections) + "\n"


def _summarize_level(level: str, items: Sequence[RunRecord]) -> LevelSummary:
    valid = [r for r in items if r.status != "error" and r.outcome is not None]
    by_scenario: dict[str, list[RunRecord]] = defaultdict(list)
    for record in valid:
        by_scenario[record.scenario_id].append(record)

    scenario_scores = [[_score(r) for r in group] for group in by_scenario.values()]
    scenario_tokens = [
        [float(r.cost.total_tokens) for r in group] for group in by_scenario.values()
    ]
    return LevelSummary(
        level=level,
        runs=len(valid),
        error_runs=len(items) - len(valid),
        step_limit_runs=sum(r.status == "step_limit" for r in valid),
        success_rate=_mean([float(_success(r)) for r in valid]),
        mean_score=_mean([_score(r) for r in valid]),
        mean_total_tokens=_mean([float(r.cost.total_tokens) for r in valid]),
        mean_tool_internal_tokens=_mean(
            [float(r.cost.tool_input_tokens + r.cost.tool_output_tokens) for r in valid]
        ),
        mean_llm_calls=_mean([float(r.cost.llm_calls) for r in valid]),
        mean_tool_calls=_mean([float(r.cost.tool_calls) for r in valid]),
        mean_wall_seconds=_mean([r.cost.wall_seconds for r in valid]),
        repeatability_score_std=_mean([_std(group) for group in scenario_scores]),
        repeatability_tokens_cv=_mean([_cv(group) for group in scenario_tokens]),
        consistency_score_std=_std([_mean(group) for group in scenario_scores]),
        consistency_tokens_cv=_cv([_mean(group) for group in scenario_tokens]),
    )


def _scenario_rows(records: Sequence[RunRecord], levels: Sequence[str]) -> list[list[str]]:
    cells: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    scenario_ids: list[str] = []
    for record in records:
        if record.scenario_id not in scenario_ids:
            scenario_ids.append(record.scenario_id)
        if record.status != "error" and record.outcome is not None:
            cells[record.scenario_id, record.level].append(record)
    rows = []
    for scenario_id in scenario_ids:
        row = [scenario_id]
        for level in levels:
            group = cells.get((scenario_id, level), [])
            if group:
                rate = _mean([float(_success(r)) for r in group])
                tokens = _mean([float(r.cost.total_tokens) for r in group])
                row.append(f"{rate:.0%} / {tokens:.0f} (n={len(group)})")
            else:
                row.append("-")
        rows.append(row)
    return rows


def _score(record: RunRecord) -> float:
    return record.outcome.score if record.outcome else 0.0


def _success(record: RunRecord) -> bool:
    return record.outcome.success if record.outcome else False


def _mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def _cv(values: Sequence[float]) -> float:
    average = _mean(values)
    return _std(values) / average if average else 0.0
