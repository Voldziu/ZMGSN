"""Aggregation of run records into per-model, per-level metrics, a Markdown report and a CSV."""

import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean, stdev

from pydantic import BaseModel, ConfigDict

from tool_eval.evaluation.analysis import (
    NON_INFERIORITY_MARGIN,
    OUTCOMES,
    Comparison,
    JudgmentIndex,
    compare_levels,
    exclusions,
    h1a,
    interactions,
    judgment_index,
    outcome,
)
from tool_eval.evaluation.checks import FACTS
from tool_eval.evaluation.models import Judgment, RunRecord
from tool_eval.evaluation.stats import pass_hat_k, wilson_interval
from tool_eval.exceptions import InvalidInputError
from tool_eval.services.formatting import format_markdown_table

PASS_K_MAX = 10
_MISSING = "-"  # undefined, or not measured by any run
_CSV_EXPANDED = {
    "pass_hat_k",
    "check_pass_rates",
    "fact_pass_rates",
}  # written as one column per k or check


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
    task_type: str = "base"  # base task or variant with an added requirement
    runs: int
    error_runs: int
    failed_runs: int
    step_limit_runs: int
    loop_runs: int  # stopped by the loop guard
    # Quality. A success is a completed run that passed every check (`RunRecord.success`).
    successes: int
    success_rate: float | None
    success_ci_low: float | None  # 95% Wilson interval
    success_ci_high: float | None
    mean_score: float | None
    check_pass_rates: dict[str, float]
    # Outcomes with the judge (see `analysis.outcome`): rate and 95% Wilson interval over the
    # runs where the outcome is decided; `judged_runs` counts runs with a judgment.
    judged_runs: int = 0
    artifact_valid_rate: float | None = None
    artifact_valid_ci_low: float | None = None
    artifact_valid_ci_high: float | None = None
    reliable_rate: float | None = None
    reliable_ci_low: float | None = None
    reliable_ci_high: float | None = None
    judged_success_rate: float | None = None
    judged_success_ci_low: float | None = None
    judged_success_ci_high: float | None = None
    mean_valid_review: float | None = None
    mean_relevance: float | None = None
    mean_faithfulness: float | None = None
    fact_pass_rates: dict[str, float] = {}
    # Stability. pass^k for k = 1..10 (index k - 1); None above the fewest runs of a scenario.
    pass_hat_k: tuple[float | None, ...]
    tool_error_rate: float | None  # tool errors per tool call
    mean_malformed_tool_calls: float | None
    malformed_rate: float | None  # malformed calls per attempted call (tool calls + malformed)
    mean_text_tool_calls: float | None  # replies holding a tool call the parser missed
    environment_error_rate: float | None  # tool errors caused by external services, per call
    agent_error_rate: float | None  # every other tool error, per call
    mean_chars_read_per_paper: float | None  # distinct characters of a paper's text read
    mean_cache_misses: float | None  # arXiv requests + PDF downloads that reached the network
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


def summarize(
    records: Sequence[RunRecord], judgments: Iterable[Judgment] = ()
) -> list[LevelSummary]:
    """Compute one summary per model and level.

    Args:
        records: Run records, possibly from several models and levels.
        judgments: The judge's ratings of these runs, if any.

    Returns:
        Summaries, base tasks before variants, then models in order of first appearance and
        levels by name (runs are stored in a random level order). Base tasks and variants are
        never pooled.
    """
    index = judgment_index(judgments)
    return [
        _summarize_group(model, level, items, task_type, index)
        for task_type, subset in _by_task_type(records).items()
        for (model, level), items in _by_model_and_level(subset).items()
    ]


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
    checks += list(dict.fromkeys(f"fact_{n}" for s in summaries for n in s.fact_pass_rates))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*base, *passes, *checks], restval="")
        writer.writeheader()
        for summary in summaries:
            row = summary.model_dump(exclude=_CSV_EXPANDED)
            row |= {f"pass_{k}": value for k, value in enumerate(summary.pass_hat_k, start=1)}
            row |= {f"check_{name}": rate for name, rate in summary.check_pass_rates.items()}
            row |= {f"fact_{name}": rate for name, rate in summary.fact_pass_rates.items()}
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def render_markdown(records: Sequence[RunRecord], judgments: Iterable[Judgment] = ()) -> str:
    """Render the full report: overview, quality, stability, cost, comparisons and run lists.

    Base tasks and variants with an added requirement get separate reports in one document;
    the variant report also breaks success down by requirement.

    Args:
        records: Run records to report on, possibly from several models.
        judgments: The judge's ratings of these runs, if any.

    Returns:
        Markdown text.
    """
    index = judgment_index(judgments)
    by_type = _by_task_type(records)
    if set(by_type) <= {"base"}:
        return _render(records, "# Evaluation report", index)
    parts = []
    if "base" in by_type:
        parts.append(_render(by_type["base"], "# Evaluation report: base tasks", index))
    variants = by_type["variant"]
    parts.append(
        _render(variants, "# Evaluation report: variants with an added requirement", index)
        + "\n## Success by requirement\n\n"
        + format_markdown_table(
            ["model", "level", "requirement", "runs", "success", "format success"],
            _requirement_rows(variants, index),
        )
        + "\n"
    )
    return "\n".join(parts)


def _requirement_rows(records: Sequence[RunRecord], index: JudgmentIndex) -> list[list[str]]:
    groups: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    for record in _scored(records):
        groups[record.model, record.level, record.requirement or _MISSING].append(record)
    rows = []
    for (model, level, requirement), items in groups.items():
        rate = sum(r.success for r in items) / len(items)
        decided = [v for r in items if (v := outcome(r, "success", index)) is not None]
        rows.append(
            [
                model,
                level,
                requirement,
                str(len(items)),
                _pct(sum(decided) / len(decided) if decided else None),
                _pct(rate),
            ]
        )
    return sorted(rows, key=lambda row: (row[0], row[1], row[2]))


def _render(records: Sequence[RunRecord], title: str, index: JudgmentIndex) -> str:
    summaries = summarize(records, index.values())
    comparisons = compare_levels(records, index)
    levels = sorted({s.level for s in summaries})
    sections = [
        title,
        *_configuration_warnings(records),
        "Metrics use the scored runs: every run except those with an infrastructure error. "
        "Two outcome axes: a run's review is *artifact valid* when every fact check passes, the "
        "judge rates it at least 4 on every scale and a variant meets its requirement; a run is "
        "*reliable* when the agent ended it itself (completed); a *success* is both. *Format "
        "success* is the original rule: completed and every format check passed. "
        f"`{_MISSING}` marks a value that is undefined or that no run measured.",
        "## Outcomes [95% Wilson interval]",
        "Rates over the runs where the outcome is decided; judged = runs with a judgment.",
        format_markdown_table(
            [
                "model",
                "level",
                "runs",
                "judged",
                "success",
                "artifact valid",
                "reliable",
                "format success",
            ],
            [
                [
                    s.model,
                    s.level,
                    str(s.runs),
                    str(s.judged_runs),
                    _rate_with_interval(
                        s.judged_success_rate, s.judged_success_ci_low, s.judged_success_ci_high
                    ),
                    _rate_with_interval(
                        s.artifact_valid_rate, s.artifact_valid_ci_low, s.artifact_valid_ci_high
                    ),
                    _rate_with_interval(s.reliable_rate, s.reliable_ci_low, s.reliable_ci_high),
                    _rate_with_interval(s.success_rate, s.success_ci_low, s.success_ci_high),
                ]
                for s in summaries
            ],
        ),
        "## Judge ratings and fact checks",
        "Mean judge ratings (1-5) and the pass rate of each fact check (ids anywhere in the file).",
        format_markdown_table(
            ["model", "level", "valid review", "relevance", "faithfulness", *FACTS],
            [
                [
                    s.model,
                    s.level,
                    _num(s.mean_valid_review, 2),
                    _num(s.mean_relevance, 2),
                    _num(s.mean_faithfulness, 2),
                    *(_pct(s.fact_pass_rates.get(name)) for name in FACTS),
                ]
                for s in summaries
            ],
        ),
        "## Overview: format success [95% Wilson interval]",
        format_markdown_table(["model", *levels], _overview_rows(summaries, levels)),
        "## Quality (format checks)",
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
                "loop",
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
                    str(s.loop_runs),
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
        "Tool error rate = tool errors per tool call, split into environment errors (network, "
        "arXiv HTTP errors, missing PDFs) and agent errors (everything else). Malformed calls "
        "(unparseable arguments, unknown tool names) never start a tool; malformed rate = "
        "malformed calls per attempted call (tool calls + malformed calls). Text calls = replies "
        "holding a tool call the server's parser missed.",
        format_markdown_table(
            [
                "model",
                "level",
                "tool calls/run",
                "tool error rate",
                "environment",
                "agent",
                "malformed/run",
                "malformed",
                "text calls/run",
            ],
            [
                [
                    s.model,
                    s.level,
                    _num(s.mean_tool_calls, 1),
                    _pct(s.tool_error_rate, 1),
                    _pct(s.environment_error_rate, 1),
                    _pct(s.agent_error_rate, 1),
                    _num(s.mean_malformed_tool_calls, 2),
                    _pct(s.malformed_rate, 1),
                    _num(s.mean_text_tool_calls, 2),
                ]
                for s in summaries
            ],
        ),
        "## Cost and time",
        "Means per scored run. Tokens include LLM calls made inside tools; tokens/success = "
        "mean tokens / success rate. Prompt overhead = input tokens of the first agent call. "
        "Times are summed call durations in seconds (parallel calls all count); other tool "
        "time is tool time outside LLM calls, mostly network. Chars/paper = distinct "
        "characters of a paper's text that reached the agent or a model; cache misses = arXiv "
        "requests and PDF downloads that reached the network.",
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
                "chars/paper",
                "cache misses",
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
                    _num(s.mean_chars_read_per_paper),
                    _num(s.mean_cache_misses, 1),
                ]
                for s in summaries
            ],
        ),
        "*Wall time depends on network and arXiv throttling; treat it as indicative only.",
        "## Level differences",
        "Later level minus earlier level within each model: outcomes as the mean of per-scenario "
        "differences in percentage points, tokens as the geometric-mean ratio, with 95% "
        "t-intervals over scenarios and exact sign-flip p-values; tokens per success as a ratio "
        "with a scenario-bootstrap BCa interval and a one-sided p-value (lower cost).",
        format_markdown_table(
            ["model", "comparison", "metric", "scenarios", "estimate", "95% CI", "p"],
            _comparison_rows(comparisons),
        ),
        *_hypothesis_sections(records, comparisons, index),
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


def _by_task_type(records: Sequence[RunRecord]) -> dict[str, list[RunRecord]]:
    """Split records into base tasks and variants, base first; empty types are left out."""
    groups: dict[str, list[RunRecord]] = {"base": [], "variant": []}
    for record in records:
        groups[record.task_type].append(record)
    return {task_type: items for task_type, items in groups.items() if items}


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


def _summarize_group(
    model: str,
    level: str,
    items: Sequence[RunRecord],
    task_type: str = "base",
    index: JudgmentIndex | None = None,
) -> LevelSummary:
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
    split = [r.cost for r in valid if r.cost.tool_errors_environment is not None]
    split_calls = sum(c.tool_calls for c in split)
    paper_reads = [n for r in valid if r.cost.chars_read for n in r.cost.chars_read.values()]
    index = index or {}
    rates = {
        name: _decided_rate([outcome(r, name, index) for r in valid])
        for name in ("success", "artifact_valid", "reliable")
    }
    judged = [j for r in valid if (j := index.get((r.config_sha256, r.run_id))) is not None]
    with_facts = [r for r in valid if r.outcome is not None and r.outcome.facts]
    return LevelSummary(
        judged_runs=len(judged),
        judged_success_rate=rates["success"][0],
        judged_success_ci_low=rates["success"][1],
        judged_success_ci_high=rates["success"][2],
        artifact_valid_rate=rates["artifact_valid"][0],
        artifact_valid_ci_low=rates["artifact_valid"][1],
        artifact_valid_ci_high=rates["artifact_valid"][2],
        reliable_rate=rates["reliable"][0],
        reliable_ci_low=rates["reliable"][1],
        reliable_ci_high=rates["reliable"][2],
        mean_valid_review=_mean(j.valid_review for j in judged),
        mean_relevance=_mean(j.relevance for j in judged),
        mean_faithfulness=_mean(f for j in judged for f in j.faithfulness),
        fact_pass_rates={
            name: sum(
                f.passed for r in with_facts if r.outcome for f in r.outcome.facts if f.name == name
            )
            / len(with_facts)
            for name in FACTS
        }
        if with_facts
        else {},
        model=model,
        level=level,
        task_type=task_type,
        runs=len(valid),
        error_runs=len(items) - len(valid),
        failed_runs=sum(r.status == "failed" for r in valid),
        step_limit_runs=sum(r.status == "step_limit" for r in valid),
        loop_runs=sum(r.status == "loop" for r in valid),
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
        mean_text_tool_calls=_mean(r.cost.text_tool_calls for r in valid),
        environment_error_rate=(
            sum(c.tool_errors_environment or 0 for c in split) / split_calls
            if split_calls
            else None
        ),
        agent_error_rate=(
            sum(c.tool_errors_agent or 0 for c in split) / split_calls if split_calls else None
        ),
        mean_chars_read_per_paper=_mean(paper_reads),
        mean_cache_misses=_mean(
            r.cost.arxiv_requests + r.cost.pdf_downloads
            for r in valid
            if r.cost.arxiv_requests is not None and r.cost.pdf_downloads is not None
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


def _decided_rate(values: Sequence[bool | None]) -> tuple[float | None, float | None, float | None]:
    decided = [v for v in values if v is not None]
    if not decided:
        return None, None, None
    interval = wilson_interval(sum(decided), len(decided))
    return (
        sum(decided) / len(decided),
        interval[0] if interval else None,
        interval[1] if interval else None,
    )


def _check_pass_rates(valid: Sequence[RunRecord]) -> dict[str, float]:
    passed: dict[str, list[bool]] = {}
    for record in valid:
        for criterion in record.outcome.criteria if record.outcome else []:
            passed.setdefault(criterion.name, []).append(criterion.passed)
    return {name: sum(values) / len(values) for name, values in passed.items()}


# --- planned comparisons ---------------------------------------------------------------


_PERCENT_METRICS = set(OUTCOMES)


def _comparison_rows(comparisons: Sequence[Comparison]) -> list[list[str]]:
    rows = []
    for c in comparisons:
        percent = c.metric in _PERCENT_METRICS
        if c.note and c.estimate is None:
            estimate, interval = c.note, _MISSING
        elif c.note:
            estimate, interval = _format_estimate(c.estimate, percent=percent), c.note
        else:
            estimate = _format_estimate(c.estimate, percent=percent)
            interval = _format_interval(c.ci, percent=percent)
        rows.append(
            [
                c.model,
                f"{c.level_b} − {c.level_a}" if percent else f"{c.level_b} / {c.level_a}",
                c.metric.replace("_", " "),
                str(c.scenarios),
                estimate,
                interval,
                _num(c.p_value, 3),
            ]
        )
    return rows


def _format_estimate(value: float | None, *, percent: bool) -> str:
    if value is None:
        return _MISSING
    return f"{value * 100:+.0f} pp" if percent else f"{value:.2f}"


def _format_interval(ci: tuple[float, float] | None, *, percent: bool) -> str:
    if ci is None:
        return _MISSING
    if percent:
        return f"{ci[0] * 100:+.0f} to {ci[1] * 100:+.0f} pp"
    return f"{ci[0]:.2f} to {ci[1]:.2f}"


def _hypothesis_sections(
    records: Sequence[RunRecord], comparisons: Sequence[Comparison], index: JudgmentIndex
) -> list[str]:
    sections: list[str] = []
    tests = h1a(comparisons)
    if tests:
        sections += [
            "## H1a: level 2 against level 1 on base tasks",
            f"Success non-inferior with a margin of {NON_INFERIORITY_MARGIN * 100:.0f} pp "
            "(one-sided sign-flip test of the shifted differences) and a lower cost per success "
            "(one-sided); p = the larger of the two, Holm-corrected across models.",
            format_markdown_table(
                [
                    "model",
                    "Δ success",
                    "95% CI",
                    "tokens/success ratio",
                    "95% CI",
                    "p",
                    "p Holm",
                    "H1a",
                ],
                [
                    [
                        t.model,
                        t.success.note or _format_estimate(t.success.estimate, percent=True),
                        _format_interval(t.success.ci, percent=True),
                        t.cost.note or _format_estimate(t.cost.estimate, percent=False),
                        _format_interval(t.cost.ci, percent=False),
                        _num(t.p_value, 3),
                        _num(t.p_holm, 3),
                        {True: "supported", False: "not supported", None: "not evaluable"}[
                            t.supported
                        ],
                    ]
                    for t in tests
                ],
            ),
        ]
    h1b = [
        c
        for c in comparisons
        if c.task_type == "variant"
        and c.metric == "success"
        and (c.level_a, c.level_b) == ("level2", "level3")
    ]
    if h1b:
        sections += [
            "## H1b: level 2 against level 3 on variants",
            "Success of level 2 minus level 3 over the variants (positive supports H1b).",
            format_markdown_table(
                ["model", "variants", "level2 − level3", "95% CI", "p"],
                [
                    [
                        c.model,
                        str(c.scenarios),
                        c.note
                        or _format_estimate(
                            None if c.estimate is None else -c.estimate, percent=True
                        ),
                        _format_interval(
                            None if c.ci is None else (-c.ci[1], -c.ci[0]), percent=True
                        ),
                        _num(c.p_value, 3),
                    ]
                    for c in h1b
                ],
            ),
        ]
    contrasts = interactions(records, index)
    if contrasts:
        sections += [
            "## Level-by-model interactions",
            "(A: level b − level a) − (B: level b − level a), success in percentage points and "
            "tokens as a log ratio, with 95% intervals from a bootstrap over shared scenarios.",
            format_markdown_table(
                ["models", "levels", "metric", "scenarios", "estimate", "95% CI"],
                [
                    [
                        f"{i.model_a} vs {i.model_b}",
                        f"{i.level_b} − {i.level_a}",
                        i.metric.replace("_", " "),
                        str(i.scenarios),
                        _format_estimate(i.estimate, percent=i.metric == "success"),
                        _format_interval(i.ci, percent=i.metric == "success"),
                    ]
                    for i in contrasts
                ],
            ),
        ]
    excluded = [e for e in exclusions(records, index) if e.excluded]
    if excluded:
        sections += [
            "## Excluded runs (sensitivity)",
            "Runs excluded as infrastructure errors, and the success rate with them counted as "
            "failures.",
            format_markdown_table(
                ["model", "level", "excluded", "success", "errors as failures"],
                [
                    [
                        e.model,
                        e.level,
                        str(e.excluded),
                        _pct(e.success_rate),
                        _pct(e.success_rate_errors_as_failures),
                    ]
                    for e in excluded
                ],
            ),
        ]
    return sections


def _rate_with_interval(rate: float | None, low: float | None, high: float | None) -> str:
    if rate is None:
        return _MISSING
    return f"{_pct(rate)} {_interval(low, high)}"


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


def _success_rate(runs: Sequence[RunRecord]) -> float:
    return sum(r.success for r in runs) / len(runs)


def _mean_tokens(runs: Sequence[RunRecord]) -> float:
    return sum(r.cost.total_tokens for r in runs) / len(runs)


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
