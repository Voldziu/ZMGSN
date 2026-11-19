"""Aggregation of run records into per-model, per-level metrics, a Markdown report and a CSV."""

import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from tool_eval.evaluation.models import RunRecord
from tool_eval.evaluation.stats import (
    BOOTSTRAP_RESAMPLES,
    cluster_bootstrap,
    pass_hat_k,
    wilson_interval,
)
from tool_eval.exceptions import InvalidInputError
from tool_eval.services.formatting import format_markdown_table

PASS_K_MAX = 10
_MISSING = "-"  # undefined, or not measured by any run
_CSV_EXPANDED = {"pass_hat_k", "check_pass_rates"}  # written as one column per k or check


class LevelSummary(BaseModel):
    """Metrics of one tool level of one model over its scored runs.

    Scored runs are all runs except those with an infrastructure `error`. `None` marks a value
    without data: every rate and mean when there is no scored run, tokens per success without
    a success, repeatability without a scenario of two scored runs, consistency with fewer than
    two scenarios, and measurements that no run made.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    level: str
    runs: int
    error_runs: int
    failed_runs: int
    step_limit_runs: int
    # Quality. A success is a completed run that passed every check (`RunRecord.success`).
    successes: int
    success_rate: float | None
    success_ci_low: float | None  # 95% Wilson interval
    success_ci_high: float | None
    mean_score: float | None
    check_pass_rates: dict[str, float]
    # Stability. pass^k for k = 1..10 (index k - 1); None above the fewest runs of a scenario.
    pass_hat_k: tuple[float | None, ...]
    tool_error_rate: float | None  # tool errors per tool call
    mean_malformed_tool_calls: float | None
    malformed_rate: float | None  # malformed calls per attempted call (tool calls + malformed)
    # Repeatability: spread across repeats of the same scenario, averaged over scenarios.
    repeatability_score_std: float | None
    repeatability_tokens_cv: float | None
    # Consistency: spread of per-scenario means across scenarios.
    consistency_score_std: float | None
    consistency_tokens_cv: float | None
    # Cost per run, including LLM calls made inside tools.
    mean_total_tokens: float | None
    mean_input_tokens: float | None
    mean_output_tokens: float | None
    mean_tool_internal_tokens: float | None
    tokens_per_success: float | None  # mean total tokens / success rate
    mean_first_call_input_tokens: float | None  # prompt overhead of the level
    mean_llm_calls: float | None
    mean_tool_calls: float | None
    # Time per run: summed call durations, and the wall clock of the agent run.
    mean_agent_llm_seconds: float | None
    mean_tool_llm_seconds: float | None
    mean_tool_other_seconds: float | None
    mean_wall_seconds: float | None


class LevelDifference(BaseModel):
    """Difference between two levels of one model, `level_b` minus `level_a`.

    Each difference is the mean, over the scenarios the two levels share, of the per-scenario
    differences, so every scenario weighs the same. The 95% percentile intervals come from a
    bootstrap that resamples those scenarios.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    level_a: str
    level_b: str
    scenarios: int
    success_rate_diff: float
    success_rate_ci: tuple[float, float]
    mean_tokens_diff: float
    mean_tokens_ci: tuple[float, float]


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


def load_results(paths: Sequence[Path]) -> list[RunRecord]:
    """Read several results files, for example one per model, into one list of records.

    Args:
        paths: JSONL files written by the runner.

    Returns:
        Records in file order.

    Raises:
        InvalidInputError: If a scored run of one model appears more than once, whatever its
            configuration: for example a file passed twice, or two results files of the same
            model. A run recorded with an infrastructure error and then scored when it was
            retried is not a duplicate.
    """
    records: list[RunRecord] = []
    seen: dict[tuple[str, str], Path] = {}
    for path in paths:
        for record in load_records(path):
            records.append(record)
            if not _is_scored(record):
                continue
            key = (record.model, record.run_id)
            if key in seen:
                msg = (
                    f"Run {record.run_id} of model {record.model} appears more than once "
                    f"({seen[key].name} and {path.name}); count each run once: do not pass a "
                    "file twice or two evaluations of the same model."
                )
                raise InvalidInputError(msg)
            seen[key] = path
    return records


def summarize(records: Sequence[RunRecord]) -> list[LevelSummary]:
    """Compute one summary per model and level.

    Args:
        records: Run records, possibly from several models and levels.

    Returns:
        Summaries, models in order of first appearance and levels by name (runs are stored in
        a random level order).
    """
    return [
        _summarize_group(model, level, items)
        for (model, level), items in _by_model_and_level(records).items()
    ]


def compare_levels(records: Sequence[RunRecord]) -> list[LevelDifference]:
    """Estimate the differences between every pair of levels of each model.

    Each difference (later level minus earlier level, by name) is the mean of the per-scenario
    differences over the scenarios both levels share, so a scenario with more scored runs at one
    level does not weigh more. Its 95% interval comes from a cluster bootstrap over those
    scenarios, because runs of one scenario are correlated; the fixed seed makes the intervals
    reproducible.

    Args:
        records: Run records, possibly from several models.

    Returns:
        One difference per model and pair of levels that share a scenario.
    """
    groups = _by_model_and_level(_scored(records))
    levels_of: dict[str, list[str]] = defaultdict(list)
    for model, level in groups:
        levels_of[model].append(level)
    differences = []
    for model, levels in levels_of.items():
        for level_a, level_b in combinations(levels, 2):
            runs_a = _by_scenario(groups[model, level_a])
            runs_b = _by_scenario(groups[model, level_b])
            per_scenario = [
                _ScenarioDifference.between(runs_a[scenario], runs_b[scenario])
                for scenario in runs_a
                if scenario in runs_b
            ]
            if not per_scenario:
                continue
            success_ci, tokens_ci = cluster_bootstrap(
                per_scenario, [_mean_success_difference, _mean_tokens_difference]
            )
            differences.append(
                LevelDifference(
                    model=model,
                    level_a=level_a,
                    level_b=level_b,
                    scenarios=len(per_scenario),
                    success_rate_diff=_mean_success_difference(per_scenario),
                    success_rate_ci=success_ci,
                    mean_tokens_diff=_mean_tokens_difference(per_scenario),
                    mean_tokens_ci=tokens_ci,
                )
            )
    return differences


def write_csv(summaries: Sequence[LevelSummary], path: Path) -> None:
    """Write one row per model and level, with every metric of the report, for plots.

    `pass_hat_k` becomes the columns `pass_1` to `pass_10`, and `check_pass_rates` one
    `check_<name>` column per check. Undefined values are empty cells.

    Args:
        summaries: Summaries from `summarize`.
        path: CSV file to write; parent directories are created.
    """
    base = [name for name in LevelSummary.model_fields if name not in _CSV_EXPANDED]
    passes = [f"pass_{k}" for k in range(1, PASS_K_MAX + 1)]
    checks = list(dict.fromkeys(f"check_{n}" for s in summaries for n in s.check_pass_rates))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*base, *passes, *checks], restval="")
        writer.writeheader()
        for summary in summaries:
            row = summary.model_dump(exclude=_CSV_EXPANDED)
            row |= {f"pass_{k}": value for k, value in enumerate(summary.pass_hat_k, start=1)}
            row |= {f"check_{name}": rate for name, rate in summary.check_pass_rates.items()}
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def render_markdown(records: Sequence[RunRecord]) -> str:
    """Render the full report: overview, quality, stability, cost, comparisons and run lists.

    Args:
        records: Run records to report on, possibly from several models.

    Returns:
        Markdown text.
    """
    summaries = summarize(records)
    levels = sorted({s.level for s in summaries})
    sections = [
        "# Evaluation report",
        *_configuration_warnings(records),
        "Metrics use the scored runs: every run except those with an infrastructure error. "
        "A run succeeds only if it completed (no step limit, not failed) and passed every check. "
        f"`{_MISSING}` marks a value that is undefined or that no run measured.",
        "## Overview: success rate [95% Wilson interval]",
        format_markdown_table(["model", *levels], _overview_rows(summaries, levels)),
        "## Quality",
        format_markdown_table(
            [
                "model",
                "level",
                "runs",
                "successes",
                "success",
                "95% CI",
                "score",
                "step limit",
                "failed",
                "errors",
            ],
            [
                [
                    s.model,
                    s.level,
                    str(s.runs),
                    str(s.successes),
                    _pct(s.success_rate),
                    _interval(s.success_ci_low, s.success_ci_high),
                    _num(s.mean_score, 2),
                    str(s.step_limit_runs),
                    str(s.failed_runs),
                    str(s.error_runs),
                ]
                for s in summaries
            ],
        ),
        "## Stability: pass^k",
        "pass^k is the probability that k repeats of a scenario all succeed: C(c,k)/C(n,k) for "
        "c successes in n scored runs, averaged over scenarios. It is undefined above the "
        "fewest scored runs of any scenario.",
        format_markdown_table(
            ["model", "level", *(f"k={k}" for k in range(1, PASS_K_MAX + 1))],
            [[s.model, s.level, *(_num(value, 2) for value in s.pass_hat_k)] for s in summaries],
        ),
        "## Tool calls",
        "Tool error rate = tool errors per tool call. Malformed calls (unparseable arguments, "
        "unknown tool names) never start a tool; malformed rate = malformed calls per attempted "
        "call (tool calls + malformed calls).",
        format_markdown_table(
            ["model", "level", "tool calls/run", "tool error rate", "malformed/run", "malformed"],
            [
                [
                    s.model,
                    s.level,
                    _num(s.mean_tool_calls, 1),
                    _pct(s.tool_error_rate, 1),
                    _num(s.mean_malformed_tool_calls, 2),
                    _pct(s.malformed_rate, 1),
                ]
                for s in summaries
            ],
        ),
        "## Cost and time",
        "Means per scored run. Tokens include LLM calls made inside tools; tokens/success = "
        "mean tokens / success rate. Prompt overhead = input tokens of the first agent call. "
        "Times are summed call durations in seconds (parallel calls all count); other tool "
        "time is tool time outside LLM calls, mostly network.",
        format_markdown_table(
            [
                "model",
                "level",
                "tokens",
                "input",
                "output",
                "tool-internal",
                "tokens/success",
                "prompt overhead",
                "LLM calls",
                "agent LLM s",
                "tool LLM s",
                "other tool s",
                "wall s*",
            ],
            [
                [
                    s.model,
                    s.level,
                    _num(s.mean_total_tokens),
                    _num(s.mean_input_tokens),
                    _num(s.mean_output_tokens),
                    _num(s.mean_tool_internal_tokens),
                    _num(s.tokens_per_success),
                    _num(s.mean_first_call_input_tokens),
                    _num(s.mean_llm_calls, 1),
                    _num(s.mean_agent_llm_seconds, 1),
                    _num(s.mean_tool_llm_seconds, 1),
                    _num(s.mean_tool_other_seconds, 1),
                    _num(s.mean_wall_seconds),
                ]
                for s in summaries
            ],
        ),
        "*Wall time depends on network and arXiv throttling; treat it as indicative only.",
        "## Level differences",
        "Later level minus earlier level within each model: the mean of the per-scenario "
        "differences over the scenarios both levels share, with 95% intervals from a cluster "
        f"bootstrap over scenarios ({BOOTSTRAP_RESAMPLES:,} resamples). `*` marks an interval "
        "that excludes zero.",
        format_markdown_table(
            ["model", "comparison", "scenarios", "Δ success", "95% CI", "Δ tokens", "95% CI"],
            _difference_rows(compare_levels(records)),
        ),
        "## Checks (pass rate)",
        _checks_table(summaries),
        "## Repeatability and consistency",
        "Repeatability = mean over scenarios of the spread across repeats "
        "(scenarios with fewer than two scored runs are skipped). "
        "Consistency = spread of per-scenario means across scenarios. "
        "`std` is of the score, `cv` is the coefficient of variation of total tokens. "
        "Lower is more stable.",
        format_markdown_table(
            [
                "model",
                "level",
                "repeat. score std",
                "repeat. tokens cv",
                "consist. score std",
                "consist. tokens cv",
            ],
            [
                [
                    s.model,
                    s.level,
                    _num(s.repeatability_score_std, 3),
                    _num(s.repeatability_tokens_cv, 2),
                    _num(s.consistency_score_std, 3),
                    _num(s.consistency_tokens_cv, 2),
                ]
                for s in summaries
            ],
        ),
        "## Per scenario (success rate / mean tokens)",
        format_markdown_table(["model", "scenario", *levels], _scenario_rows(records, levels)),
    ]
    failed = [r for r in records if r.status == "failed"]
    if failed:
        lines = "\n".join(f"- `{r.model}/{r.run_id}`: {r.error}" for r in failed)
        sections += ["## Runs that failed inside the agent (scored as failures)", lines]
    errors = [r for r in records if r.status == "error"]
    if errors:
        lines = "\n".join(f"- `{r.model}/{r.run_id}`: {r.error}" for r in errors)
        sections += [
            "## Runs with infrastructure errors (excluded from all metrics)",
            lines,
        ]
    return "\n\n".join(sections) + "\n"


# --- grouping and per-group metrics ----------------------------------------------------


def _is_scored(record: RunRecord) -> bool:
    return record.status != "error" and record.outcome is not None


def _scored(records: Iterable[RunRecord]) -> list[RunRecord]:
    return [r for r in records if _is_scored(r)]


def _by_model_and_level(records: Sequence[RunRecord]) -> dict[tuple[str, str], list[RunRecord]]:
    """Group records by model (first appearance) and level (by name)."""
    model_order = {
        model: index for index, model in enumerate(dict.fromkeys(r.model for r in records))
    }
    groups: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        groups[record.model, record.level].append(record)
    return {key: groups[key] for key in sorted(groups, key=lambda k: (model_order[k[0]], k[1]))}


def _by_scenario(records: Iterable[RunRecord]) -> dict[str, list[RunRecord]]:
    groups: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        groups[record.scenario_id].append(record)
    return dict(groups)


def _summarize_group(model: str, level: str, items: Sequence[RunRecord]) -> LevelSummary:
    valid = _scored(items)
    by_scenario = _by_scenario(valid)
    scenario_scores = [[_score(r) for r in group] for group in by_scenario.values()]
    scenario_tokens = [
        [float(r.cost.total_tokens) for r in group] for group in by_scenario.values()
    ]
    successes = sum(r.success for r in valid)
    success_rate = successes / len(valid) if valid else None
    interval = wilson_interval(successes, len(valid))
    counts = [(sum(r.success for r in group), len(group)) for group in by_scenario.values()]
    tool_calls = sum(r.cost.tool_calls for r in valid)
    calls_and_malformed = [
        (r.cost.tool_calls, malformed)
        for r in valid
        if (malformed := r.cost.malformed_tool_calls) is not None
    ]
    attempted = sum(calls + malformed for calls, malformed in calls_and_malformed)
    mean_total_tokens = _mean(r.cost.total_tokens for r in valid)
    return LevelSummary(
        model=model,
        level=level,
        runs=len(valid),
        error_runs=len(items) - len(valid),
        failed_runs=sum(r.status == "failed" for r in valid),
        step_limit_runs=sum(r.status == "step_limit" for r in valid),
        successes=successes,
        success_rate=success_rate,
        success_ci_low=interval[0] if interval else None,
        success_ci_high=interval[1] if interval else None,
        mean_score=_mean(_score(r) for r in valid),
        check_pass_rates=_check_pass_rates(valid),
        pass_hat_k=tuple(pass_hat_k(counts, k) for k in range(1, PASS_K_MAX + 1)),
        tool_error_rate=sum(r.cost.tool_errors for r in valid) / tool_calls if tool_calls else None,
        mean_malformed_tool_calls=_mean(malformed for _, malformed in calls_and_malformed),
        malformed_rate=(
            sum(malformed for _, malformed in calls_and_malformed) / attempted
            if attempted
            else None
        ),
        repeatability_score_std=_mean(_std(g) for g in scenario_scores),
        repeatability_tokens_cv=_mean(_cv(g) for g in scenario_tokens),
        consistency_score_std=_std([mean(group) for group in scenario_scores]),
        consistency_tokens_cv=_cv([mean(group) for group in scenario_tokens]),
        mean_total_tokens=mean_total_tokens,
        mean_input_tokens=_mean(
            r.cost.agent_input_tokens + r.cost.tool_input_tokens for r in valid
        ),
        mean_output_tokens=_mean(
            r.cost.agent_output_tokens + r.cost.tool_output_tokens for r in valid
        ),
        mean_tool_internal_tokens=_mean(
            r.cost.tool_input_tokens + r.cost.tool_output_tokens for r in valid
        ),
        tokens_per_success=(
            mean_total_tokens / success_rate
            if mean_total_tokens is not None and success_rate
            else None
        ),
        mean_first_call_input_tokens=_mean(r.cost.first_call_input_tokens for r in valid),
        mean_llm_calls=_mean(r.cost.llm_calls for r in valid),
        mean_tool_calls=_mean(r.cost.tool_calls for r in valid),
        mean_agent_llm_seconds=_mean(r.cost.agent_llm_seconds for r in valid),
        mean_tool_llm_seconds=_mean(r.cost.tool_llm_seconds for r in valid),
        mean_tool_other_seconds=_mean(r.cost.tool_other_seconds for r in valid),
        mean_wall_seconds=_mean(r.cost.wall_seconds for r in valid),
    )


def _check_pass_rates(valid: Sequence[RunRecord]) -> dict[str, float]:
    passed: dict[str, list[bool]] = {}
    for record in valid:
        for criterion in record.outcome.criteria if record.outcome else []:
            passed.setdefault(criterion.name, []).append(criterion.passed)
    return {name: sum(values) / len(values) for name, values in passed.items()}


# --- level differences -------------------------------------------------------------------


class _ScenarioDifference(NamedTuple):
    """One scenario's difference between two levels (b minus a): success rate, mean tokens."""

    success: float
    tokens: float

    @classmethod
    def between(
        cls, runs_a: Sequence[RunRecord], runs_b: Sequence[RunRecord]
    ) -> "_ScenarioDifference":
        return cls(
            success=_success_rate(runs_b) - _success_rate(runs_a),
            tokens=_mean_tokens(runs_b) - _mean_tokens(runs_a),
        )


def _success_rate(runs: Sequence[RunRecord]) -> float:
    return sum(r.success for r in runs) / len(runs)


def _mean_tokens(runs: Sequence[RunRecord]) -> float:
    return sum(r.cost.total_tokens for r in runs) / len(runs)


# Plain sums, not statistics.mean: these run 10,000 times per comparison.
def _mean_success_difference(sample: Sequence[_ScenarioDifference]) -> float:
    return sum(d.success for d in sample) / len(sample)


def _mean_tokens_difference(sample: Sequence[_ScenarioDifference]) -> float:
    return sum(d.tokens for d in sample) / len(sample)


def _difference_rows(differences: Sequence[LevelDifference]) -> list[list[str]]:
    return [
        [
            d.model,
            f"{d.level_b} − {d.level_a}",
            str(d.scenarios),
            f"{d.success_rate_diff * 100:+.0f} pp",
            _signed_interval(d.success_rate_ci, scale=100, unit=" pp"),
            f"{d.mean_tokens_diff:+.0f}",
            _signed_interval(d.mean_tokens_ci),
        ]
        for d in differences
    ]


def _signed_interval(interval: tuple[float, float], *, scale: float = 1, unit: str = "") -> str:
    low, high = interval
    mark = " *" if low > 0 or high < 0 else ""
    return f"{low * scale:+.0f} to {high * scale:+.0f}{unit}{mark}"


# --- tables ---------------------------------------------------------------------------------


def _configuration_warnings(records: Sequence[RunRecord]) -> list[str]:
    configs: dict[str, set[str]] = defaultdict(set)
    for record in _scored(records):
        configs[record.model].add(record.config_sha256)
    return [
        f"**Warning:** the runs of `{model}` come from {len(hashes)} different configurations; "
        "its levels are not comparable."
        for model, hashes in configs.items()
        if len(hashes) > 1
    ]


def _overview_rows(summaries: Sequence[LevelSummary], levels: Sequence[str]) -> list[list[str]]:
    cells = {(s.model, s.level): s for s in summaries}
    rows = []
    for model in dict.fromkeys(s.model for s in summaries):
        row = [model]
        for level in levels:
            summary = cells.get((model, level))
            if summary is None or summary.success_rate is None:
                row.append(_MISSING)
            else:
                interval = _interval(summary.success_ci_low, summary.success_ci_high)
                row.append(f"{_pct(summary.success_rate)} [{interval}] n={summary.runs}")
        rows.append(row)
    return rows


def _checks_table(summaries: Sequence[LevelSummary]) -> str:
    names = list(dict.fromkeys(name for s in summaries for name in s.check_pass_rates))
    return format_markdown_table(
        ["model", "level", *names],
        [
            [s.model, s.level, *(_pct(s.check_pass_rates.get(name)) for name in names)]
            for s in summaries
        ],
    )


def _scenario_rows(records: Sequence[RunRecord], levels: Sequence[str]) -> list[list[str]]:
    cells: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    keys: dict[tuple[str, str], None] = {}
    for record in records:
        keys.setdefault((record.model, record.scenario_id))
        if _is_scored(record):
            cells[record.model, record.scenario_id, record.level].append(record)
    rows = []
    for model, scenario_id in keys:
        row = [model, scenario_id]
        for level in levels:
            group = cells.get((model, scenario_id, level), [])
            if group:
                rate = _success_rate(group)
                row.append(f"{rate:.0%} / {_mean_tokens(group):.0f} (n={len(group)})")
            else:
                row.append(_MISSING)
        rows.append(row)
    return rows


# --- small helpers -------------------------------------------------------------------------


def _score(record: RunRecord) -> float:
    return record.outcome.score if record.outcome else 0.0


def _pct(value: float | None, digits: int = 0) -> str:
    return _MISSING if value is None else f"{value:.{digits}%}"


def _num(value: float | None, digits: int = 0) -> str:
    return _MISSING if value is None else f"{value:.{digits}f}"


def _interval(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return _MISSING
    return f"{low * 100:.0f}–{high * 100:.0f}%"


def _mean(values: Iterable[float | None]) -> float | None:
    """Return the mean of the known values, or `None` if no value is known."""
    known = [float(value) for value in values if value is not None]
    return mean(known) if known else None


def _std(values: Sequence[float]) -> float | None:
    return stdev(values) if len(values) >= 2 else None


def _cv(values: Sequence[float]) -> float | None:
    spread = _std(values)
    average = mean(values) if values else 0.0
    return spread / average if spread is not None and average else None
