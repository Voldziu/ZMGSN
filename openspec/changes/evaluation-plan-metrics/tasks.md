# Tasks

## 1. Success rule

- [x] 1.1 Add the computed field `RunRecord.success` (`completed` and every check passed). Document `OutcomeScore.success` as "every check passed". Use `RunRecord.success` in the report, the MLflow `success` metric and the runner's log line. Verify with tests:
  - a completed run that passes every check succeeds;
  - a `step_limit` run that passes every check does not succeed and keeps score 1.0;
  - a `failed` run keeps its partial score;
  - a record line without `success` (the old format) and with status `step_limit` reads as not successful;
  - new JSONL lines contain `success`.
- [x] 1.2 State the success rule in the README and verify that the README sentence matches the `review-scoring` spec.

## 2. Meter: time, prompt overhead and malformed tool calls

- [x] 2.1 Add the optional `CostMetrics` fields `agent_llm_seconds`, `tool_llm_seconds`, `tool_seconds`, `first_call_input_tokens`, `invalid_tool_calls` and `unknown_tool_calls` (each `None` by default), and the derived `tool_other_seconds` and `malformed_tool_calls`. Verify with a test that a record without these fields loads with `None` values and unchanged tokens and calls.
- [x] 2.2 Time model and tool calls in `RunMeter` with an injectable clock. Verify with a fake-clock test: two agent calls of 3 s give 6 s, and a 60 s tool with 40 s of internal model calls gives tool 60, tool model 40 and other 20. Verify with the real agent-loop test that the times are non-negative and that tool time is at least tool model time.
- [x] 2.3 Record the input tokens of the first agent call. Verify with tests:
  - first call 100 and second call 150 give 100;
  - a first call without usage gives `None`;
  - tool-internal calls are never the first call.
- [x] 2.4 Count malformed tool calls from agent replies: `invalid_tool_calls`, and names missing from `RunMeter(tool_names=…)`. Verify with a `create_agent` loop driven by a fake model whose replies call an unknown tool and hold an unparseable call:
  - both counts are 1;
  - `tool_calls` and `tool_errors` exclude them;
  - a run stopped by the step limit keeps the counts;
  - a meter without tool names reports `unknown_tool_calls` as `None`.
- [x] 2.5 Verify with a new agent-loop test that a tool call failing validation counts one tool call and one tool error (behaviour that the new `run-metering` spec now states).

## 3. Runner: snapshot before scoring and level order

- [x] 3.1 In `run_single`, create the meter with the level's tool names and take the snapshot before `evaluate_review`. Verify with a runner test in which scoring advances the meter's clock by 30 s: the recorded wall time excludes those 30 s, and the record holds the timing fields.
- [x] 3.2 Add `block_order(levels, seed, repeat, scenario_id)` and loop repeat → scenario → `block_order`. Make the existing order-dependent runner and tracing tests independent of the order. Verify with tests:
  - the same inputs give the same order, whatever order the levels are listed in;
  - every block holds all levels;
  - with a fixed seed, ten blocks do not all share one order;
  - repeat 1 finishes before repeat 2 starts.
- [x] 3.3 Add `order_seed` and `block_position` to `RunRecord` and to the MLflow parameters. Resolve the seed: the explicit value, else the last one recorded in the output file, else a random draw. Add `run --order-seed`. Verify with tests:
  - an explicit seed is stored in every record;
  - a resumed evaluation reuses the recorded seed;
  - a new file draws and records a seed;
  - the recorded positions match `block_order`;
  - the configuration hash is the same for different seeds;
  - the CLI passes `--order-seed` through.
- [x] 3.4 Document the new measurements, the level order and `--order-seed` in the README, and verify that the documented command line matches `main.py`.

## 4. MLflow

- [x] 4.1 Log the known new cost fields and the derived `total_tokens`, `llm_calls`, `malformed_tool_calls` and `tool_other_seconds` as metrics, never `None`. Log `order_seed` and `block_position` as parameters. Update the README's MLflow paragraph. Verify with the fake-MLflow tests:
  - the metrics include total tokens, first-call input tokens and the three times;
  - no metric is `None`;
  - the parameters include the order seed and the block position;
  - a `step_limit` run that passed every check logs `success` 0 and `score` 1.

## 5. Statistics module

- [x] 5.1 Add `evaluation/stats.py` with `wilson_interval`, `pass_hat_k`, `percentile` and `cluster_bootstrap`. Verify with unit tests:
  - Wilson bounds 8/10 → (0.490, 0.943), 10/10 → (0.722, 1.0), 0/10 → (0.0, 0.278), and `None` for n = 0;
  - pass^k for the two-scenario example of the spec (0.75, 0.611, 0.5), and `None` when k exceeds a scenario's runs;
  - percentile interpolation;
  - the bootstrap is reproducible for a fixed seed, gives (0, 0) for constant data, and covers the point estimate in a simple case.

## 6. Report

- [x] 6.1 Summarise per model and level: `LevelSummary.model`, models by first appearance and levels by name. Add successes with the Wilson interval, pass^1..10, input and output tokens, tokens per success, the tool-error and malformed-call rates, the first-call tokens, the time means skipping unknown values, and the per-check pass rates. Verify with tests on constructed records, including mixed old and new records and a model and level without any success.
- [x] 6.2 Add the level differences: within each model and for each level pair (later − earlier), on the shared scenarios, with cluster-bootstrap intervals for success rate and tokens; mark intervals that exclude zero. Verify with tests:
  - identical levels give 0 with (0, 0) and no mark;
  - level 2 succeeding where level 1 never does, on 10 scenarios, gives a marked interval above zero;
  - a scenario missing at one level is left out;
  - two renderings are identical.
- [x] 6.3 Add `load_results(paths)`, which rejects a scored run appearing twice, and make the configuration warning per model. Verify with tests:
  - the same file passed twice fails with an error that names the run;
  - an `error` record followed by its retry is accepted;
  - two models with one configuration each give no warning;
  - one model with two configurations gives a warning that names it.
- [x] 6.4 Lay out the Markdown report:
  1. the model × level overview;
  2. Quality, Stability, Tool calls, Cost and time;
  3. Level differences, Checks;
  4. Repeatability and consistency, Per scenario;
  5. the run lists.

  Add the CSV export and the CLI `report FILE... [--csv PATH]`. Verify with CLI tests:
  - two files give one row per model and level;
  - the CSV header holds `success_ci_low`, `success_ci_high`, `pass_1`…`pass_10` and `tokens_per_success`;
  - an undefined value is an empty cell;
  - the Markdown is still printed.
- [x] 6.5 Make `eval-presets.sh` print the report's `## Quality` section instead of `## Per level`, and document the report and CSV in the README. Verify with a test that runs the script's `awk` extraction on a rendered report and gets the quality table only.

## 7. Verification

- [x] 7.1 Run `make test` (pytest, ruff, mypy) and `uv run ruff format --check .`, and verify that all of them pass.
- [x] 7.2 Render the report and CSV from the local smoke results files (written before this change) and from a synthetic set of three models × three levels × ten scenarios × ten repeats. Check by eye that:
  - old records load and show unknown values as `-`;
  - the tables are complete;
  - the report takes at most a few seconds.

## 8. Review fixes

- [x] 8.1 Compute each level difference as the mean of per-scenario differences over the shared scenarios, for success rate and for tokens, and bootstrap those differences. Verify with a test in which both levels have the same rate in every scenario but level 2 has only 2 scored runs in one of them: the difference is 0 with (0, 0). The existing difference tests must still pass.
- [x] 8.2 Make undefined values `None`:
  - the rates and means of a model and level without scored runs;
  - repeatability without a scenario of two scored runs;
  - consistency with fewer than two scenarios.

  Remove `_if_scored`. Verify with tests: a level with only error runs shows `-` in the Markdown and empty cells in the CSV for success rate, score and means; a single repeat leaves repeatability undefined.
- [x] 8.3 Plan every block over all levels of the tool set, run the given levels in that order, and record positions in that plan. Make `order_seed` and `block_position` required in `run_single`. Verify with tests:
  - resuming a level-1-only results file with all three levels gives unique positions per block that match `block_order` over the tool set;
  - a run with only levels 1 and 3 keeps their planned relative order;
  - a retried error run keeps its planned position.
- [x] 8.4 Create the meter after the run's MLflow run is open. Verify with a fake-MLflow test in which opening the run advances the meter's clock by 10 s: the recorded wall time excludes those 10 s.
- [x] 8.5 Clean up and verify with `make test`:
  - reuse `_scored()` throughout the report;
  - merge `_pct` and `_rate`;
  - rename `percentile` to `quantile`, the malformed-call pairs, and the test helper `_rec` to `_record_for`;
  - drop the unused `z`, `confidence` and `compare_levels(resamples, seed)` parameters;
  - name files, not paths, in the duplicate-run error.
- [x] 8.6 Update the README: the block plan over the tool set, undefined values, the duplicate-run error. Verify that the README matches the code.
- [ ] 8.7 Run `make test`, `uv run ruff format --check .` and `openspec validate evaluation-plan-metrics --strict`. Render the report from the local `qwen-weak` and `qwen-mid` run on the Mac (`rag-2024h1`, three levels, two repeats). Check the records for:
  - model and tool time, and first-call input tokens that differ by level;
  - malformed-call counts;
  - order seeds and positions that match `block_order`.

## Workflow follow-up

- On the RTX host, run one evaluation against a real preset (for example `make eval-presets PRESETS=qwen REPEATS=1 ONLY= LEVELS="level1 level2 level3"`). Check that the records hold model and tool time, first-call input tokens that differ by level, malformed-call counts, order seeds and block positions, and that MLflow shows the new metrics.
- Update `docs/overleaf/methods.tex`: malformed calls are now counted, and wall time excludes scoring.
- Archive the change after review. This creates `run-metering`, `run-order` and `evaluation-report`, and updates `review-scoring` and `run-tracing`.
