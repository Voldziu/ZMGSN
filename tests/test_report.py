import csv
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import pytest

from test_evaluation import _record
from tool_eval.evaluation.models import CriterionResult, OutcomeScore, RunRecord, RunStatus
from tool_eval.evaluation.report import (
    compare_levels,
    load_results,
    render_markdown,
    summarize,
    write_csv,
)
from tool_eval.evaluation.stats import (
    Z_95,
    cluster_bootstrap,
    pass_hat_k,
    quantile,
    wilson_interval,
)
from tool_eval.exceptions import InvalidInputError
from tool_eval.main import main

# --- statistics -------------------------------------------------------------------


def test_wilson_interval() -> None:
    assert pytest.approx(1.959964, abs=1e-6) == Z_95
    assert wilson_interval(8, 10) == pytest.approx((0.4902, 0.9433), abs=1e-4)
    # Exact at the boundaries, not 0.9999999999999999.
    all_succeeded = wilson_interval(10, 10)
    assert all_succeeded is not None
    assert all_succeeded[0] == pytest.approx(0.7225, abs=1e-4)
    assert all_succeeded[1] == 1.0
    none_succeeded = wilson_interval(0, 10)
    assert none_succeeded is not None
    assert none_succeeded[0] == 0.0
    assert none_succeeded[1] == pytest.approx(0.2775, abs=1e-4)
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError, match="between 0 and n"):
        wilson_interval(11, 10)


def test_pass_hat_k() -> None:
    counts = [(10, 10), (5, 10)]
    assert pass_hat_k(counts, 1) == pytest.approx(0.75)
    assert pass_hat_k(counts, 2) == pytest.approx(0.6111, abs=1e-4)
    assert pass_hat_k(counts, 10) == pytest.approx(0.5)
    # A scenario with only 3 runs bounds k.
    short = [(3, 3), (5, 10)]
    assert pass_hat_k(short, 3) == pytest.approx((1 + 10 / 120) / 2)
    assert pass_hat_k(short, 4) is None
    assert pass_hat_k([], 1) is None
    with pytest.raises(ValueError, match="at least 1"):
        pass_hat_k(counts, 0)


def test_quantile_interpolates_between_ranks() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert quantile(values, 0) == 1.0
    assert quantile(values, 1) == 4.0
    assert quantile(values, 0.5) == 2.5
    assert quantile(values, 0.25) == pytest.approx(1.75)
    assert quantile([7.0], 0.975) == 7.0
    with pytest.raises(ValueError):
        quantile([], 0.5)


def _mean(sample: Sequence[float]) -> float:
    return mean(sample)


def test_cluster_bootstrap_is_reproducible_and_covers_the_estimate() -> None:
    clusters = [float(i) for i in range(10)]
    [interval] = cluster_bootstrap(clusters, [_mean], resamples=2000, seed=1)
    assert interval == cluster_bootstrap(clusters, [_mean], resamples=2000, seed=1)[0]
    low, high = interval
    assert low < 4.5 < high
    assert low > 0
    assert high < 9


def test_cluster_bootstrap_of_constant_data_is_a_point() -> None:
    [first, second] = cluster_bootstrap([3.0] * 5, [_mean, lambda s: 0.0], resamples=100)
    assert first == (3.0, 3.0)
    assert second == (0.0, 0.0)
    with pytest.raises(ValueError, match="zero clusters"):
        cluster_bootstrap([], [_mean])


# --- summaries -----------------------------------------------------------------------


def _record_for(
    level: str,
    scenario: str,
    repeat: int = 1,
    *,
    model: str = "qwen",
    passed: bool = True,
    status: RunStatus = "completed",
    tokens: int = 1000,
    config: str = "config-1",
    cost: dict[str, Any] | None = None,
    checks: Sequence[bool] | None = None,
) -> RunRecord:
    base = _record(
        level, scenario, repeat, score=1.0 if passed else 0.5, tokens=tokens, status=status
    )
    update: dict[str, Any] = {"model": model, "config_sha256": config}
    if cost:
        update["cost"] = base.cost.model_copy(update=cost)
    if checks is not None:
        update["outcome"] = OutcomeScore.from_criteria(
            [CriterionResult(name=f"check{i}", passed=ok) for i, ok in enumerate(checks)]
        )
    return base.model_copy(update=update)


def test_summaries_per_model_and_level() -> None:
    more_tokens = {"agent_output_tokens": 500, "tool_input_tokens": 300, "tool_output_tokens": 200}
    records = [
        *(
            _record_for("level2", f"s{i}", tokens=39_000, passed=i < 5, cost=more_tokens)
            for i in range(10)
        ),
        *(_record_for("level1", f"s{i}", tokens=10, passed=i < 8) for i in range(10)),
        _record_for("level1", "s0", model="llama", passed=False),
        _record_for("level1", "s1", model="llama", status="error"),
    ]
    level1, level2, llama = summarize(records)
    assert [(s.model, s.level) for s in (level1, level2, llama)] == [
        ("qwen", "level1"),
        ("qwen", "level2"),
        ("llama", "level1"),
    ]
    assert (level1.runs, level1.successes, level1.success_rate) == (10, 8, 0.8)
    assert level1.success_ci_low == pytest.approx(0.4902, abs=1e-4)
    assert level1.success_ci_high == pytest.approx(0.9433, abs=1e-4)
    assert level2.mean_total_tokens == pytest.approx(40_000)
    assert (level2.mean_input_tokens, level2.mean_output_tokens) == (39_300, 700)
    assert level2.mean_tool_internal_tokens == 500
    assert level2.tokens_per_success == pytest.approx(80_000)
    assert (llama.runs, llama.error_runs, llama.successes) == (1, 1, 0)
    assert llama.tokens_per_success is None
    assert llama.success_ci_low == 0.0


def test_level_with_only_errors_has_undefined_rates_and_means(tmp_path: Path) -> None:
    records = [_record_for("level1", "a", status="error"), _record_for("level2", "a")]
    errors_only, _ = summarize(records)
    assert (errors_only.runs, errors_only.error_runs) == (0, 1)
    for name in (
        "success_rate",
        "mean_score",
        "mean_total_tokens",
        "mean_input_tokens",
        "mean_output_tokens",
        "mean_tool_internal_tokens",
        "mean_llm_calls",
        "mean_tool_calls",
        "mean_wall_seconds",
        "repeatability_score_std",
        "consistency_score_std",
    ):
        assert getattr(errors_only, name) is None, name

    text = render_markdown(records)
    [quality] = [line for line in text.splitlines() if line.startswith("| qwen | level1 | 0 | 0 |")]
    assert quality == "| qwen | level1 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1 |"
    assert "| qwen | - | 100% [21–100%] n=1 |" in text

    out = tmp_path / "summary.csv"
    write_csv(summarize(records), out)
    with out.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["level"] == "level1"
    for column in ("success_rate", "mean_score", "mean_total_tokens", "mean_wall_seconds"):
        assert row[column] == "", column


def test_repeatability_and_consistency_need_enough_runs() -> None:
    [single] = summarize([_record_for("level1", f"s{i}") for i in range(3)])
    assert single.repeatability_score_std is None
    assert single.repeatability_tokens_cv is None
    assert single.consistency_score_std == 0.0
    [one_scenario] = summarize([_record_for("level1", "s0", r) for r in (1, 2)])
    assert one_scenario.repeatability_score_std == 0.0
    assert one_scenario.consistency_score_std is None
    assert one_scenario.consistency_tokens_cv is None


def test_pass_hat_k_per_level() -> None:
    records = [
        *(_record_for("level1", "easy", r) for r in range(1, 11)),
        *(_record_for("level1", "hard", r, passed=r <= 5) for r in range(1, 11)),
        *(_record_for("level2", "easy", r) for r in range(1, 11)),
        *(_record_for("level2", "hard", r) for r in range(1, 4)),
    ]
    level1, level2 = summarize(records)
    assert level1.pass_hat_k[0] == pytest.approx(0.75)
    assert level1.pass_hat_k[1] == pytest.approx(0.6111, abs=1e-4)
    assert level1.pass_hat_k[9] == pytest.approx(0.5)
    assert len(level1.pass_hat_k) == 10
    # "hard" has only 3 runs at level 2.
    assert level2.pass_hat_k[:3] == (1.0, 1.0, 1.0)
    assert level2.pass_hat_k[3:] == (None,) * 7


def test_tool_call_rates_time_and_older_records() -> None:
    new = {
        "tool_calls": 10,
        "tool_errors": 1,
        "invalid_tool_calls": 0,
        "unknown_tool_calls": 0,
        "agent_llm_seconds": 4.0,
        "tool_llm_seconds": 2.0,
        "tool_seconds": 5.0,
        "first_call_input_tokens": 900,
    }
    records = [
        _record_for("level1", "a", 1, cost=new | {"invalid_tool_calls": 1}),
        _record_for("level1", "a", 2, cost=new),
        # Written before time, malformed calls and the first call were measured.
        _record_for("level1", "a", 3, cost={"tool_calls": 0, "tool_errors": 0}),
    ]
    [summary] = summarize(records)
    assert summary.tool_error_rate == pytest.approx(0.1)
    assert summary.malformed_rate == pytest.approx(1 / 21)
    assert summary.mean_malformed_tool_calls == pytest.approx(0.5)
    assert summary.mean_agent_llm_seconds == pytest.approx(4.0)
    assert summary.mean_tool_other_seconds == pytest.approx(3.0)
    assert summary.mean_first_call_input_tokens == pytest.approx(900)
    text = render_markdown(records)
    assert "10.0%" in text
    assert "4.8%" in text

    [old] = summarize(records[2:])
    assert old.mean_agent_llm_seconds is None
    assert old.malformed_rate is None
    assert old.tool_error_rate is None


def test_check_pass_rates() -> None:
    records = [
        _record_for("level1", "a", 1, checks=[True, True]),
        _record_for("level1", "a", 2, checks=[True, False]),
    ]
    [summary] = summarize(records)
    assert summary.check_pass_rates == {"check0": 1.0, "check1": 0.5}
    assert "| qwen | level1 | 100% | 50% |" in render_markdown(records)


# --- level differences ------------------------------------------------------------------


def test_level_differences_are_paired_by_scenario_and_marked() -> None:
    records = []
    for i in range(10):
        for repeat in (1, 2):
            records.append(_record_for("level1", f"s{i}", repeat, passed=False, tokens=100))
            records.append(_record_for("level2", f"s{i}", repeat, tokens=100 + 10 * i))
    # Only at level 1, so left out of the comparison.
    records.append(_record_for("level1", "level1-only", tokens=10**6))

    [difference] = compare_levels(records)
    assert (difference.level_a, difference.level_b, difference.scenarios) == (
        "level1",
        "level2",
        10,
    )
    assert difference.success_rate_diff == 1.0
    assert difference.success_rate_ci == (1.0, 1.0)
    assert difference.mean_tokens_diff == pytest.approx(45)
    low, high = difference.mean_tokens_ci
    assert 0 < low < 45 < high
    text = render_markdown(records)
    assert "| qwen | level2 − level1 | 10 | +100 pp | +100 to +100 pp * | +45 |" in text
    assert render_markdown(records) == text  # the bootstrap is seeded


def test_identical_levels_differ_by_zero_without_a_mark() -> None:
    records = [
        _record_for(level, f"s{i}", tokens=500) for level in ("level1", "level3") for i in range(5)
    ]
    [difference] = compare_levels(records)
    assert difference.success_rate_diff == 0.0
    assert difference.success_rate_ci == (0.0, 0.0)
    assert difference.mean_tokens_ci == (0.0, 0.0)
    [row] = [line for line in render_markdown(records).splitlines() if "level3 − level1" in line]
    assert "+0 to +0 pp" in row
    assert "*" not in row


def test_level_differences_weigh_every_scenario_the_same() -> None:
    # The same rate in every scenario, but level 2 has 8 unretried errors in "easy".
    records = [
        *(_record_for("level1", "hard", r, passed=False) for r in range(1, 11)),
        *(_record_for("level1", "easy", r) for r in range(1, 11)),
        *(_record_for("level2", "hard", r, passed=False) for r in range(1, 11)),
        *(_record_for("level2", "easy", r) for r in range(1, 3)),
        *(_record_for("level2", "easy", r, status="error") for r in range(3, 11)),
    ]
    [difference] = compare_levels(records)
    assert difference.scenarios == 2
    assert difference.success_rate_diff == 0.0
    assert difference.success_rate_ci == (0.0, 0.0)
    assert difference.mean_tokens_diff == 0.0


def test_differences_stay_within_each_model() -> None:
    records = [
        _record_for(level, "a", model=model)
        for model in ("qwen", "llama")
        for level in ("level1", "level2")
    ]
    assert [(d.model, d.level_a, d.level_b) for d in compare_levels(records)] == [
        ("qwen", "level1", "level2"),
        ("llama", "level1", "level2"),
    ]


# --- inputs and warnings ------------------------------------------------------------------


def _write(path: Path, records: Sequence[RunRecord]) -> Path:
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    return path


def test_results_files_combine_but_each_run_counts_once(tmp_path: Path) -> None:
    qwen = _write(tmp_path / "qwen.jsonl", [_record_for("level1", "a")])
    llama = _write(tmp_path / "llama.jsonl", [_record_for("level1", "a", model="llama")])
    assert len(load_results([qwen, llama])) == 2
    with pytest.raises(InvalidInputError, match="level1-a-01 of model qwen") as error:
        load_results([qwen, llama, qwen])
    # Files are named, not given as absolute paths (see tool_eval.exceptions).
    assert "(qwen.jsonl and qwen.jsonl)" in str(error.value)
    assert str(tmp_path) not in str(error.value)

    retried = _write(
        tmp_path / "retried.jsonl",
        [_record_for("level1", "a", status="error"), _record_for("level1", "a")],
    )
    [summary] = summarize(load_results([retried]))
    assert (summary.runs, summary.error_runs) == (1, 1)


def test_configuration_warning_is_per_model() -> None:
    two_models = [
        _record_for("level1", "a", config="qwen-config"),
        _record_for("level1", "a", model="llama", config="llama-config"),
    ]
    assert "Warning" not in render_markdown(two_models)
    mixed = [_record_for("level1", "a", config="one"), _record_for("level2", "a", config="two")]
    assert (
        "**Warning:** the runs of `qwen` come from 2 different configurations"
        in render_markdown(mixed)
    )


# --- Markdown and CSV ----------------------------------------------------------------------


def test_cli_reports_several_files_and_writes_a_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    qwen = _write(
        tmp_path / "qwen.jsonl",
        [
            _record_for(level, f"s{i}", passed=i < 8)
            for level in ("level2", "level1")
            for i in range(10)
        ],
    )
    llama = _write(
        tmp_path / "llama.jsonl",
        [
            _record_for("level1", "s0", model="llama", passed=False),
            _record_for("level2", "s0", model="llama"),
        ],
    )
    out = tmp_path / "plots" / "summary.csv"
    assert main(["report", str(qwen), str(llama), "--csv", str(out)]) == 0

    text = capsys.readouterr().out
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Overview: success rate [95% Wilson interval]",
        "## Quality",
        "## Stability: pass^k",
        "## Tool calls",
        "## Cost and time",
        "## Level differences",
        "## Checks (pass rate)",
        "## Repeatability and consistency",
        "## Per scenario (success rate / mean tokens)",
    ]
    assert "| model | level1 | level2 |" in text
    assert "| qwen | 80% [49–94%] n=10 | 80% [49–94%] n=10 |" in text

    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["model"], r["level"]) for r in rows] == [
        ("qwen", "level1"),
        ("qwen", "level2"),
        ("llama", "level1"),
        ("llama", "level2"),
    ]
    columns = set(rows[0])
    assert {"success_ci_low", "success_ci_high", "tokens_per_success", "check_x"} <= columns
    assert {f"pass_{k}" for k in range(1, 11)} <= columns
    assert float(rows[0]["success_ci_low"]) == pytest.approx(0.4902, abs=1e-4)
    assert rows[2]["tokens_per_success"] == ""  # llama level 1 never succeeded
    assert rows[2]["pass_2"] == ""  # a single run per scenario

    assert main(["report", str(qwen), str(qwen)]) == 2


def test_new_tool_and_reading_measures_are_summarized() -> None:
    from tool_eval.evaluation.report import summarize

    record = _record("level1", "a", 1, score=1.0, tokens=10)
    cost = record.cost.model_copy(
        update={
            "tool_calls": 10,
            "tool_errors": 3,
            "tool_errors_environment": 1,
            "tool_errors_agent": 2,
            "text_tool_calls": 1,
            "chars_read": {"2401.00001": 6000, "2401.00002": 18000},
            "arxiv_requests": 2,
            "pdf_downloads": 1,
        }
    )
    old = _record("level1", "b", 1, score=1.0, tokens=10)
    summary = summarize([record.model_copy(update={"cost": cost}), old])[0]
    assert summary.environment_error_rate == 0.1
    assert summary.agent_error_rate == 0.2
    assert summary.mean_text_tool_calls == 1
    assert summary.mean_chars_read_per_paper == 12000
    assert summary.mean_cache_misses == 3
