"""Planned comparisons of the evaluation: level differences, H1a, H1b and interactions.

Every comparison pairs scenarios: a level difference is the mean over scenarios of the
per-scenario difference between two levels of one model and task type, with a 95% t-interval
(df = scenarios - 1) and an exact sign-flip permutation p-value. Tokens are compared as
geometric-mean ratios (differences of mean log tokens). Tokens per success are compared as a
ratio with a scenario-cluster BCa interval. Comparisons where both levels are at 100% (or 0%)
in every scenario are labelled instead of given a zero-width interval.

Outcomes of a run (see `RunRecord`):
- `success`: artifact valid (facts, judge, requirement) and reliable (completed);
- `artifact_valid`, `reliable`;
- `format_success`: completed and every format check passed (the original rule).
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations

from pydantic import BaseModel, ConfigDict

from tool_eval.evaluation.models import Judgment, RunRecord
from tool_eval.evaluation.stats import (
    bca_interval,
    cluster_bootstrap,
    holm,
    paired_t_interval,
    sign_flip_p,
)

# H1a's non-inferiority margin for success, level 2 minus level 1 (fixed before the runs).
NON_INFERIORITY_MARGIN = 0.15
ALPHA = 0.05

JudgmentIndex = Mapping[tuple[str, str], Judgment]
OUTCOMES = ("success", "artifact_valid", "reliable", "format_success")


def judgment_index(judgments: Iterable[Judgment]) -> dict[tuple[str, str], Judgment]:
    """Judgments by (configuration hash, run id); later judgments replace earlier ones."""
    return {(j.config_sha256, j.run_id): j for j in judgments}


def outcome(record: RunRecord, name: str, judgments: JudgmentIndex) -> bool | None:
    """Value of one outcome for a run; `None` when it cannot be decided yet.

    Args:
        record: The run.
        name: One of `OUTCOMES`.
        judgments: Judge ratings by (configuration hash, run id).

    Returns:
        True, False or None.
    """
    judgment = judgments.get((record.config_sha256, record.run_id))
    if name == "success":
        return record.success_with(judgment)
    if name == "artifact_valid":
        return record.artifact_valid(judgment)
    if name == "reliable":
        return record.reliable
    return record.format_success


class Comparison(BaseModel):
    """One planned comparison between two levels of one model and task type."""

    model_config = ConfigDict(frozen=True)

    model: str
    task_type: str
    level_a: str
    level_b: str
    metric: str  # an outcome, `tokens`, `tokens_completed` or `tokens_per_success`
    scenarios: int
    estimate: float | None  # b - a for outcomes; b / a ratio for tokens
    ci: tuple[float, float] | None
    p_value: float | None
    note: str = ""  # `both at ceiling`, `both at floor` or `not evaluable`
    # Per-scenario differences (b - a for outcomes, log ratios for tokens) behind the estimate.
    differences: list[float] = []


def compare_levels(records: Sequence[RunRecord], judgments: JudgmentIndex) -> list[Comparison]:
    """Every level pair of every model and task type, for each outcome and the cost measures.

    Args:
        records: Scored and unscored runs; runs with infrastructure errors are left out.
        judgments: Judge ratings by (configuration hash, run id).

    Returns:
        Comparisons, `level_b` minus (or over) `level_a` with levels in name order.
    """
    cells = _cells(records)
    comparisons: list[Comparison] = []
    keys = sorted({(model, task) for model, task, _ in cells})
    for model, task in keys:
        levels = sorted(level for m, t, level in cells if (m, t) == (model, task))
        for level_a, level_b in combinations(levels, 2):
            a, b = cells[model, task, level_a], cells[model, task, level_b]
            shared = sorted(set(a) & set(b))
            if not shared:
                continue
            base = {"model": model, "task_type": task, "level_a": level_a, "level_b": level_b}
            for name in OUTCOMES:
                comparisons.append(
                    _outcome_comparison(base, name, [(a[s], b[s]) for s in shared], judgments)
                )
            for metric, only_completed in (("tokens", False), ("tokens_completed", True)):
                comparisons.append(
                    _token_comparison(base, metric, [(a[s], b[s]) for s in shared], only_completed)
                )
            comparisons.append(_tps_comparison(base, [(a[s], b[s]) for s in shared], judgments))
    return comparisons


class H1aResult(BaseModel):
    """H1a for one model: level 2 non-inferior to level 1 in success and cheaper per success."""

    model_config = ConfigDict(frozen=True)

    model: str
    success: Comparison
    cost: Comparison
    p_value: float | None  # max of the one-sided p-values (intersection-union test)
    p_holm: float | None
    supported: bool | None


def h1a(comparisons: Sequence[Comparison]) -> list[H1aResult]:
    """Test H1a per model on base tasks, with Holm correction across models.

    Success is non-inferior when the lower bound of `level2 - level1` lies above
    `-NON_INFERIORITY_MARGIN`; its one-sided p-value comes from the sign-flip test of the
    shifted differences. Cost is lower when the tokens-per-success ratio's one-sided p-value is
    below alpha. H1a holds when both hold after Holm correction.

    Args:
        comparisons: Output of `compare_levels`.

    Returns:
        One result per model with base runs at levels 1 and 2.
    """
    found = {
        (c.model, c.metric): c
        for c in comparisons
        if c.task_type == "base" and (c.level_a, c.level_b) == ("level1", "level2")
    }
    models = sorted({model for model, _ in found})
    rows = []
    for model in models:
        success, cost = found.get((model, "success")), found.get((model, "tokens_per_success"))
        if success is None or cost is None:
            continue
        p_values = [
            _non_inferiority_p(success),
            cost.p_value if cost.estimate is not None and cost.estimate < 1 else None,
        ]
        p = None if None in p_values else max(p for p in p_values if p is not None)
        rows.append((model, success, cost, p))
    adjusted = holm([p for *_, p in rows])
    return [
        H1aResult(
            model=model,
            success=success,
            cost=cost,
            p_value=p,
            p_holm=p_adj,
            supported=None if p_adj is None else p_adj <= ALPHA,
        )
        for (model, success, cost, p), p_adj in zip(rows, adjusted, strict=True)
    ]


class Interaction(BaseModel):
    """Difference of two models' level differences, `(A_b - A_a) - (B_b - B_a)`."""

    model_config = ConfigDict(frozen=True)

    model_a: str
    model_b: str
    task_type: str
    level_a: str
    level_b: str
    metric: str  # `success` (risk difference) or `log_tokens`
    scenarios: int
    estimate: float
    ci: tuple[float, float]


def interactions(
    records: Sequence[RunRecord], judgments: JudgmentIndex, *, seed: int = 0
) -> list[Interaction]:
    """Level-by-model interactions, resampling scenarios jointly across both models.

    Args:
        records: Runs.
        judgments: Judge ratings.
        seed: Seed of the bootstrap.

    Returns:
        One interaction per model pair, level pair, task type and metric with shared scenarios.
    """
    cells = _cells(records)
    results = []
    by_task: dict[str, set[str]] = defaultdict(set)
    for model, task, _ in cells:
        by_task[task].add(model)
    for task, models in sorted(by_task.items()):
        levels = sorted({level for _, t, level in cells if t == task})
        for model_a, model_b in combinations(sorted(models), 2):
            for level_a, level_b in combinations(levels, 2):
                quads = [
                    cells.get((m, task, lv))
                    for m in (model_a, model_b)
                    for lv in (level_a, level_b)
                ]
                if any(q is None for q in quads):
                    continue
                groups = [q for q in quads if q is not None]
                shared = sorted(set.intersection(*(set(g) for g in groups)))
                if len(shared) < 2:
                    continue
                for metric, value in (
                    ("success", lambda runs: _rate(runs, "success", judgments)),
                    ("log_tokens", _mean_log_tokens),
                ):
                    contrasts = []
                    for s in shared:
                        values = [value(g[s]) for g in groups]
                        if None in values:
                            continue
                        aa, ab, ba, bb = (v for v in values if v is not None)
                        contrasts.append((ab - aa) - (bb - ba))
                    if len(contrasts) < 2:
                        continue
                    (ci,) = cluster_bootstrap(contrasts, [_mean], seed=seed)
                    results.append(
                        Interaction(
                            model_a=model_a,
                            model_b=model_b,
                            task_type=task,
                            level_a=level_a,
                            level_b=level_b,
                            metric=metric,
                            scenarios=len(contrasts),
                            estimate=_mean(contrasts),
                            ci=ci,
                        )
                    )
    return results


class Exclusions(BaseModel):
    """Runs excluded as infrastructure errors, and success with them counted as failures."""

    model_config = ConfigDict(frozen=True)

    model: str
    task_type: str
    level: str
    excluded: int
    success_rate: float | None
    success_rate_errors_as_failures: float | None


def exclusions(records: Sequence[RunRecord], judgments: JudgmentIndex) -> list[Exclusions]:
    """Excluded runs and the sensitivity of the success rate, per model, task type and level."""
    groups: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        groups[record.model, record.task_type, record.level].append(record)
    rows = []
    for (model, task, level), items in sorted(groups.items()):
        scored = [r for r in items if r.status != "error" and r.outcome is not None]
        values = [outcome(r, "success", judgments) for r in scored]
        decided = [v for v in values if v is not None]
        errors = len(items) - len(scored)
        rows.append(
            Exclusions(
                model=model,
                task_type=task,
                level=level,
                excluded=errors,
                success_rate=sum(decided) / len(decided) if decided else None,
                success_rate_errors_as_failures=(
                    sum(decided) / (len(decided) + errors) if decided or errors else None
                ),
            )
        )
    return rows


# --- helpers --------------------------------------------------------------------------------


Cells = dict[tuple[str, str, str], dict[str, list[RunRecord]]]


def _cells(records: Sequence[RunRecord]) -> Cells:
    """Scored runs by (model, task type, level), then by scenario."""
    cells: Cells = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.status == "error" or record.outcome is None:
            continue
        cells[record.model, record.task_type, record.level][record.scenario_id].append(record)
    return cells


def _rate(runs: Sequence[RunRecord], name: str, judgments: JudgmentIndex) -> float | None:
    values = [outcome(r, name, judgments) for r in runs]
    decided = [v for v in values if v is not None]
    return sum(decided) / len(decided) if decided else None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _mean_log_tokens(runs: Sequence[RunRecord]) -> float | None:
    tokens = [r.cost.total_tokens for r in runs if r.cost.total_tokens > 0]
    return _mean([math.log(t) for t in tokens]) if tokens else None


def _outcome_comparison(
    base: Mapping[str, str],
    name: str,
    pairs: Sequence[tuple[list[RunRecord], list[RunRecord]]],
    judgments: JudgmentIndex,
) -> Comparison:
    rates = [(_rate(a, name, judgments), _rate(b, name, judgments)) for a, b in pairs]
    defined = [(ra, rb) for ra, rb in rates if ra is not None and rb is not None]
    diffs = [rb - ra for ra, rb in defined]
    note = ""
    if defined and all(ra == rb == 1 for ra, rb in defined):
        note = "both at ceiling"
    elif defined and all(ra == rb == 0 for ra, rb in defined):
        note = "both at floor"
    interval = paired_t_interval(diffs) if not note else None
    return Comparison(
        **base,
        metric=name,
        scenarios=len(diffs),
        estimate=_mean(diffs) if diffs else None,
        ci=(interval[1], interval[2]) if interval else None,
        p_value=sign_flip_p(diffs) if diffs and not note else None,
        note=note or ("" if diffs else "not evaluable"),
        differences=diffs,
    )


def _token_comparison(
    base: Mapping[str, str],
    metric: str,
    pairs: Sequence[tuple[list[RunRecord], list[RunRecord]]],
    only_completed: bool,
) -> Comparison:
    diffs = []
    for runs_a, runs_b in pairs:
        if only_completed:
            runs_a = [r for r in runs_a if r.reliable]
            runs_b = [r for r in runs_b if r.reliable]
        la, lb = _mean_log_tokens(runs_a), _mean_log_tokens(runs_b)
        if la is not None and lb is not None:
            diffs.append(lb - la)
    interval = paired_t_interval(diffs)
    return Comparison(
        **base,
        metric=metric,
        scenarios=len(diffs),
        estimate=math.exp(_mean(diffs)) if diffs else None,
        ci=(math.exp(interval[1]), math.exp(interval[2])) if interval else None,
        p_value=sign_flip_p(diffs) if diffs else None,
        note="" if diffs else "not evaluable",
        differences=diffs,
    )


def _tps_comparison(
    base: Mapping[str, str],
    pairs: Sequence[tuple[list[RunRecord], list[RunRecord]]],
    judgments: JudgmentIndex,
) -> Comparison:
    def totals(runs: Sequence[RunRecord]) -> tuple[float, int]:
        decided = [(r, outcome(r, "success", judgments)) for r in runs]
        tokens = sum(r.cost.total_tokens for r, v in decided if v is not None)
        return float(tokens), sum(1 for _, v in decided if v)

    clusters = [(totals(a), totals(b)) for a, b in pairs]

    def ratio(sample: Sequence[tuple[tuple[float, int], tuple[float, int]]]) -> float | None:
        tokens_a = sum(c[0][0] for c in sample)
        successes_a = sum(c[0][1] for c in sample)
        tokens_b = sum(c[1][0] for c in sample)
        successes_b = sum(c[1][1] for c in sample)
        if not successes_a or not successes_b or not tokens_a:
            return None
        return (tokens_b / successes_b) / (tokens_a / successes_a)

    estimate = ratio(clusters)
    if estimate is None:
        return Comparison(
            **base,
            metric="tokens_per_success",
            scenarios=len(clusters),
            estimate=None,
            ci=None,
            p_value=None,
            note="not evaluable",
        )
    per_scenario = [
        math.log((b[0] / b[1]) / (a[0] / a[1])) for a, b in clusters if a[1] and b[1] and a[0]
    ]
    return Comparison(
        **base,
        metric="tokens_per_success",
        scenarios=len(clusters),
        estimate=estimate,
        ci=bca_interval(clusters, ratio),
        p_value=sign_flip_p(per_scenario, alternative="less") if len(per_scenario) >= 2 else None,
    )


def _non_inferiority_p(success: Comparison) -> float | None:
    """One-sided sign-flip p-value that `level2 - level1` lies above `-NON_INFERIORITY_MARGIN`."""
    if not success.differences:
        return None
    shifted = [d + NON_INFERIORITY_MARGIN for d in success.differences]
    return sign_flip_p(shifted, alternative="greater")
