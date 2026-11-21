# Proposal

## Why

On 2026-11-19 the group accepted an evaluation plan with four axes: quality, stability (pass^k), cost and flexibility. It also decided that a run which hits the step limit is never a success. The paper's methods (`methods.tex`: "Outcome evaluation", "Cost measurement") and its plan (`experiment_plan.tex`: "Design", "Metrics and analysis") now promise measurements and statistics that the code does not produce yet:

- **Success ignores the run status.** `OutcomeScore.success` only says that every check passed, and the report and MLflow use it as is. A `step_limit` or `failed` run whose review passes all checks therefore counts as a success.
- **Time is one number that includes scoring.** `RunMeter` reports only `wall_seconds`, and the runner takes its snapshot after `evaluate_review`, which looks up arXiv (with throttling). LLM time is not separated from tool and network time.
- **The prompt-overhead covariate is not recorded.** The tool schemas differ in size between levels (the confound in `experiment_plan.tex`), but nothing stores the input tokens of the first agent call.
- **Malformed tool calls are invisible.** Unparseable calls (`AIMessage.invalid_tool_calls`) and calls to tool names the level does not have never start a tool run, so `on_tool_start` never counts them.
- **Levels always run in the given order.** `run_evaluation` loops repeat → scenario → level, so within every block the first level pays for cold PDF and arXiv caches.
- **The report cannot carry the analysis.** It reads one file, groups only by level, and has no confidence intervals, pass^k, tokens per success, level-difference intervals or machine-readable output for plots.

All of this must exist before the main experiment (about 3,000 runs) starts, because those runs are too expensive to repeat. It is the classes 8–9 milestone item "implement the evaluation methods and record quality, time, cost and tokens".

## What Changes

- **Success rule (BREAKING for reported numbers):** a run is a success only when its status is `completed` and every check passes. `step_limit` and `failed` runs keep their score as partial credit. Each record gets a top-level `success`. The report applies the rule to older results files too.
- **Time:** the meter sums the duration of agent LLM calls, of tool-internal LLM calls and of tool calls. The remaining tool time is derived from them. The runner takes the cost snapshot before scoring, so wall time no longer includes it.
- **Prompt overhead:** the meter records the input tokens of the first agent LLM call.
- **Malformed tool calls:** the meter counts the agent's unparseable tool calls and its calls to tool names that the level does not have. Neither counts as a tool call.
- **Storage:** the new measurements are stored in the results file and logged to MLflow. Records written before this change show them as unknown (`null`), not as zero.
- **Level order:** each (repeat, scenario) block has a random planned order over all levels of the tool set, derived from an order seed, and runs its levels in that order. Each record stores the seed and the run's position in the plan, so positions stay unique when a file is resumed with other levels. `run --order-seed` fixes the seed; a resumed evaluation reuses the seed recorded in its results file.
- **Report:**
  - `report` accepts several results files and groups the runs by model and level;
  - it adds the success rate with a 95% Wilson interval, pass^k for k = 1..10, tokens per success, per-check pass rates, tool-error and malformed-call rates, and LLM and tool time;
  - it estimates the level differences in success rate and tokens within each model as means of per-scenario differences, with 95% intervals from a cluster bootstrap over scenarios (10,000 resamples);
  - values without data, such as the success rate of a level with only error runs, are undefined rather than zero;
  - `--csv` writes one row per model and level for plots;
  - it rejects a run that appears twice, for example when a file is passed twice.
- **`make eval-presets`:** after each preset it prints the new quality table instead of the old per-level table.

## Capabilities

### New Capabilities

- `run-metering`: what the meter measures during each run: model calls and tokens split between the agent and tools, tool calls and errors, malformed tool calls, LLM and tool time, wall time, and the input tokens of the first agent call.
- `run-order`: the order in which an evaluation executes its runs: a randomised level order within each (repeat, scenario) block, derived from a recorded seed.
- `evaluation-report`: how results files are aggregated into the report and the CSV export: grouping by model and level, interval estimates, pass^k, cost per success and level comparisons.

### Modified Capabilities

- `review-scoring`: adds the rule that a run succeeds only when it completed and passed every check.
- `run-tracing`: the MLflow run also carries the order seed and block position as parameters. Its metrics add the new time measurements, the first call's input tokens, malformed tool calls and total tokens, and `success` follows the success rule.

## Impact

- **Code:**
  - `src/tool_eval/evaluation/`: `models.py`, `cost.py`, `runner.py`, `report.py`, `tracing.py`, and a new `stats.py` for the interval estimates;
  - `src/tool_eval/main.py` (`run --order-seed`; `report` with several files and `--csv`);
  - `src/tool_eval/server/eval-presets.sh`.
- **Docs:** README (success rule, new measurements, level order, report).
- **Results format:**
  - new optional record fields: `success`, `order_seed`, `block_position`, and the new cost fields;
  - the configuration hash does not change, so existing results files can still be resumed and reported;
  - wall time of new records excludes scoring.
- **Reported numbers:** `step_limit` and `failed` runs that passed every check no longer count as successes. The pilot had none.
- **Dependencies:** none. The statistics use the Python standard library.
- **Paper (outside this change):** `methods.tex` says that malformed calls "go uncounted so far" and that wall time includes scoring. Both statements become outdated.
