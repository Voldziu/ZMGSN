# Design

## Context

See proposal.md for the motivation and the specs for the required behaviour. Current state that shapes the approach:

- **Meter.** `evaluation/cost.py::RunMeter` is a LangChain callback handler passed to `run_task`.
  - It marks every tool run in `on_tool_start` and attributes a model call to a tool when the call's parent run is a tool run.
  - Tokens come from `usage_metadata` in `on_llm_end`; tool errors come from `on_tool_error`.
  - `snapshot()` returns `CostMetrics` with `wall_seconds` since the meter was created.
- **Runner.** `runner.run_single` runs the agent, then scores the review with `evaluate_review`, and only then calls `meter.snapshot()`. Scoring looks up arXiv through the cache, with requests spaced 3 s apart.
- **Malformed tool calls in LangChain 1.4.4 / LangGraph 1.2.14.**
  - **Unparseable arguments.** `langchain-openai` puts a tool call whose `arguments` are not valid JSON into `AIMessage.invalid_tool_calls`. `create_agent` routes only on `tool_calls`, so a reply with nothing but invalid calls ends the loop. Mixed replies get an error `ToolMessage` for each invalid call on the next model call (`_patch_invalid_tool_calls`).
  - **Unknown tool names.** LangGraph's `ToolNode._validate_tool_call` answers a call to an unknown name with "Error: … is not a valid tool" and never invokes a tool, so no `on_tool_start` fires.
  - **Lost messages.** On `step_limit` and `failed` runs, `run_task` raises, so the message history is lost.
- **Success.** `OutcomeScore.success` means that all criteria passed. The report (`_success`), the MLflow metric (`tracing._log_run`) and the runner's log line all use it as the run's success.
- **Order.** `_execute_runs` loops repeat → scenario → level, in the order the levels were given.
- **Report.**
  - `report.py` groups one file's records by level (`LevelSummary`), and `main.py report` takes one path.
  - `server/eval-presets.sh` cuts the `## Per level` section out of the report.
  - The configuration warning fires when a file holds more than one `config_sha256`. The hash covers the model, so it would fire for every report of several models.
- **Resuming results files.** The configuration hash does not change with this change, so `make eval-presets REPEATS=10` resumes a pilot results file. One file can therefore hold records written before and after the change.
- **Dependencies.** numpy is not installed, and nothing else in the project needs it.

## Goals / Non-Goals

**Goals:**
- Every measurement and statistic promised in the paper's "Outcome evaluation", "Cost measurement", "Design" and "Metrics and analysis" sections, except the task variants and frozen caches.
- Measurements that do not depend on MLflow: the results file stays the source of the paper's numbers.
- Reports that are reproducible: the same files always give the same intervals.

**Non-Goals:**
- **Task variants and frozen data caches:** separate changes.
- **Other statistical models:** no mixed-effects models or other inference beyond what the plan states.
- **Plots:** the CSV feeds them; the plotting itself is not part of this change.
- **Scoring time:** it is not recorded separately; it is simply left out of wall time.
- **Malformed calls in plain text:** tool calls that vLLM's parser did not recognise at all arrive as ordinary text and are not detected.

## Decisions

### Success is a computed field of the run record
- **The rule.** `RunRecord.success` is `status == "completed" and outcome is not None and outcome.success`.
- **Computed field.** It is a pydantic computed field. It is written into every new JSONL line and recomputed whenever a record is read, so a stored value never overrides the rule and older files follow it too.
- **Existing field.** `OutcomeScore.success` keeps its name and its meaning, "every check passed", so existing files load unchanged. Its docstring points to `RunRecord.success`.
- **Users.** The report, the MLflow `success` metric and the runner's log line use `RunRecord.success`.
- **Typing.** mypy needs `# type: ignore[prop-decorator]` on the computed field, the idiom pydantic documents.

*Alternatives:*
- **Decide success in `evaluate_review`:** rejected. The checks do not know the run status, and records written earlier would keep the old meaning.
- **Rename `OutcomeScore.success` to `all_passed`:** clearer, but it breaks every reader of existing results files unless an alias is kept. The top-level `success` field removes the ambiguity at lower cost.

### Time comes from paired callbacks in the meter
- **Model calls.** `on_chat_model_start` stores a start time per run ID. `on_llm_end` and `on_llm_error` add the elapsed time to `agent_llm_seconds` or `tool_llm_seconds`, using the same attribution as the tokens. A failed model call adds time but, as before, is not counted as a call.
- **Tool calls.** `on_tool_start` stores a start time. `on_tool_end` and `on_tool_error` add the elapsed time to `tool_seconds`, which includes the model calls made inside the tool.
- **Remaining tool time.** `CostMetrics.tool_other_seconds = max(0, tool_seconds − tool_llm_seconds)`. It is the time tools spend on arXiv and PDF downloads, PDF parsing and files, which is dominated by network access.
- **Overlap.** Overlapping calls are added: parallel tool calls run in `ToolNode`'s thread pool. The sums measure time spent in calls; wall time stays the elapsed time.
- **Testing.** `RunMeter(clock=…)` takes an injectable clock (default `time.perf_counter`), so tests can check exact durations.

*Alternatives:*
- **MLflow span durations:** rejected, because tracing is optional.
- **`wrap_model_call` / `wrap_tool_call` middleware:** agent middleware does not see the model calls made inside tools, and callbacks do.

### Snapshot before scoring
- `run_single` takes `cost = meter.snapshot()` as soon as the agent loop has ended and its status is known, before `evaluate_review`.
- It creates the meter only after the run's MLflow run and root span are open, just before it builds the agent. Wall time then covers building the agent and the agent loop, and nothing that depends on the tracking server.

### Prompt overhead is the server's input count for the first agent call
- On the first agent `on_llm_end`, the meter stores the sum of the reported input tokens. Later calls never change it. If that first call reports no usage, the value stays `None`: a later call's input includes the history, so it cannot stand in.
- **What it covers.** The first prompt is the system prompt, the task and the tool schemas, rendered by the model's chat template. That is the overhead in the unit used for cost.
- **Per-level covariate.** The task text differs slightly between scenarios (topic and dates), so the covariate is the per-level mean.

*Alternative:* tokenising the schemas on the client. Rejected: it needs every model's tokenizer and chat template, while the server's count is exact and free.

### Malformed tool calls are counted at the model's reply
- **Where.** In `on_llm_end` for agent-side calls, the meter inspects the returned `AIMessage`:
  - each entry of `invalid_tool_calls` adds to `invalid_tool_calls`;
  - each entry of `tool_calls` whose name the level does not offer adds to `unknown_tool_calls`.
- **Tool names.** The runner passes the level's tool names (`get_tools(level)`) to the meter. A meter created without tool names leaves `unknown_tool_calls` unknown.
- **Why at the reply:**
  - one place covers both kinds;
  - it does not depend on how LangGraph routes them;
  - the counts survive `step_limit` and `failed` runs, whose message history is lost.
- **Tool calls unchanged.** Malformed calls are not tool calls or tool errors, so `tool_calls` keeps meaning tool runs. Attempted calls are tool calls plus malformed calls.

*Alternatives:*
- **Scanning the final message history:** it loses exactly the runs that end at the step limit or fail.
- **A `wrap_tool_call` middleware that checks `request.tool is None`:** it sees only unknown names, and it can reach the meter only through custom callback events.

### New measurements are optional, and `None` means unknown
- **New fields.** `CostMetrics` gains `agent_llm_seconds`, `tool_llm_seconds`, `tool_seconds`, `first_call_input_tokens`, `invalid_tool_calls` and `unknown_tool_calls`, each `… | None = None`.
- **Values.** The meter fills them (0 when nothing happened). The exceptions are first-call tokens without usage, and unknown names when the meter has no tool names.
- **Older records.** Records written before the change load with `None`, so means skip them instead of averaging in zeros. Files that mix old and new records are expected, because the configuration hash is unchanged.
- **MLflow.** MLflow logs only known values, because its metrics must be numbers. It also logs the derived `total_tokens`, `llm_calls`, `malformed_tool_calls` and `tool_other_seconds`.

*Alternative:* defaults of 0. Rejected: they would silently bias the means of mixed files.

### Level order is a seeded permutation per block
- **The permutation.** `runner.block_order(levels, seed=…, repeat=…, scenario_id=…)` seeds `random.Random` with the string `"{seed}:{repeat}:{scenario_id}"`. It draws one `random()` key per level, in sorted level order, and sorts the levels by their keys.
  - String seeds are hashed with SHA-512, independent of `PYTHONHASHSEED`.
  - Python guarantees that `random()` sequences stay reproducible across versions. `shuffle` has no such guarantee, so the keys come from `random()` alone.
  - The order depends on the set of levels, not on the order the user listed them in.
- **Plan over the whole tool set.** The runner plans every block over all levels of the tool set: `block_order([*TOOLSETS, *levels], …)`. It runs the levels it was given in that order.
  - **Unique positions.** A results file can be resumed with other `--levels` (the Makefile default is `LEVELS=level1`). Positions within a block still stay unique, and the plan is recomputable from the seed alone.
  - **Subsets.** A subset of the levels keeps its relative order, which is still random, but its positions can have gaps.
- **Loop.** The runner loops repeat → scenario → the planned order. Repeats stay outermost, so an interrupted evaluation keeps balanced data.
- **Record.** `RunRecord` gains `order_seed` and `block_position`, 1-based in the planned order over the tool set. `run_single` requires both and logs them as MLflow parameters. Like the sampling seed, the order seed is not part of `config_fingerprint`.
  - A retried run keeps its planned position. The order in which runs actually ran is in `started_at`, which every record already has.
- **Seed resolution.** `run_evaluation(order_seed=None)` uses, in this order:
  1. the explicit value (`run --order-seed`);
  2. the last `order_seed` recorded in the output file;
  3. a new `secrets.randbelow(2**31)`.

  The seed is logged at the start.

*Alternatives:*
- **Shuffling all (scenario, level) pairs of a repeat:** it breaks the blocks and adds nothing.
- **Counterbalancing the levels (a Latin square):** 3! orders do not divide 10 repeats evenly, and the paper states randomisation. The recorded positions let the analysis check for position effects.
- **Planning only the given levels:** positions would depend on `--levels`. A file resumed with other levels would mix two plans, and a block could hold two runs at position 1 (found in review).
- **Recording the realised position instead of the planned one:** it needs the finished runs of every block, and a resume still mixes orders. `started_at` already records the realised order.

### Report: summaries per model and level, statistics in a pure module
- **Input.**
  - `main.py report` takes `nargs="*"` files (default `results/results.jsonl`) and `--csv PATH`.
  - `report.load_results(paths)` concatenates the files.
  - It raises `InvalidInputError` when a scored run with the same `(model, run_id)` appears twice, whatever the configuration. An `error` record followed by its retry is not a duplicate.
  - The error names the run and the two files by name, not by path, because domain error messages carry no absolute paths (`exceptions.py`).
  - Two configurations of one model that cover the same runs are therefore rejected, not only warned about: counting both would count those runs twice. The warning covers models whose configurations hold different runs.
- **Grouping.** `summarize(records)` returns one `LevelSummary` (now with a `model` field) per model and level. Models follow first appearance; levels are sorted by name, because the file order is now random.
- **New module `evaluation/stats.py`.** It holds pure functions that use only the standard library (`math.comb`, `random`):
  - `wilson_interval(successes, n)`. It uses z = 1.959963984540054 and is exact at 0 and n.
  - `pass_hat_k(counts, k)`, with `(c, n)` per scenario. It is `None` when some scenario has `n < k`.
  - `quantile` and `cluster_bootstrap(clusters, statistics, resamples=10_000, seed=0)`. It draws the clusters with replacement from `random.Random(seed)` and returns one 95% percentile interval per statistic. All statistics share the same draws.
  - No other parameters (the z of the Wilson interval, the confidence level, or the resamples and seed of a comparison): nothing needs them.
- **Metrics:**
  - **Wilson interval.** It treats the runs as independent. It describes each model and level, as the paper states; the bootstrap carries the clustering for comparisons.
  - **pass^k.** For each scenario with c successes in n scored runs, `comb(c, k) / comb(n, k)` is the unbiased estimator of the probability that k repeats all succeed (τ-bench). The report averages it over scenarios for k = 1..10.
  - **Tokens per success.** The mean total tokens divided by the success rate, or `None` when the rate is 0.
  - **Tool-error rate:** Σ tool errors / Σ tool calls.
  - **Malformed-call rate:** Σ malformed / Σ (tool calls + malformed).
  - **Pooled rates.** Both rates are pooled over runs: each is the probability that one call fails, so runs with many calls weigh more.
  - **Means of the new measurements.** They skip `None` values.
- **Level differences.**
  - For each model and each level pair a < b (by name), the report estimates b − a on the scenarios that both levels have.
  - **Statistics.** Each shared scenario s contributes two per-scenario differences:
    - its success rate at b minus its success rate at a;
    - its mean tokens at b minus its mean tokens at a.

    The estimate is the mean of these differences over the scenarios, so every scenario weighs the same.
  - **Why not pooled rates.** Pooled rates (all successes of b over all runs of b, minus the same for a) let the scenario mix leak into the difference when the levels have different numbers of runs in a scenario, for example unretried errors or an interrupted block. The review reproduced −33 pp for two levels with identical rates in every scenario. With balanced data, the two estimators agree.
  - **Bootstrap.** 10,000 resamples of the scenarios with a fixed seed, and 2.5/97.5 percentiles with linear interpolation.
  - **Pairing.** Pairing by scenario keeps the scenario effect out of the difference.
  - **Marking.** An interval that excludes zero is marked `*`.
- **Markdown layout:**
  1. an overview with success rate [Wilson interval] per model × level;
  2. Quality;
  3. Stability (pass^k);
  4. Tool calls;
  5. Cost and time;
  6. Level differences;
  7. Checks;
  8. Repeatability and consistency (kept);
  9. Per scenario (kept, with a model column);
  10. the failed and error run lists, which name the model.
- **Configuration warning.** It becomes per model.
- **Undefined values.** Some values have no data behind them, and their summary fields are `None`:
  - the rates and means of a model and level without scored runs;
  - repeatability when no scenario has two scored runs;
  - consistency when fewer than two scenarios have scored runs.

  The Markdown shows `-` and the CSV an empty cell, so plots never read a missing value as zero. Before this, a level with only error runs got `0.0`, which the review flagged.
- **CSV.** `csv.DictWriter` writes one row per `LevelSummary`. The lists are flattened to `pass_1..pass_10` and `check_<name>`, and `None` becomes an empty cell.
- **`eval-presets.sh`.** It prints the `## Quality` section, cut by heading with `awk`, instead of the removed `## Per level`.

*Alternatives:*
- **Newcombe intervals for differences:** they ignore the correlation of runs within a scenario.
- **Mixed-effects logistic regression:** it needs statsmodels and modelling choices the plan does not state.
- **BCa bootstrap:** more accurate with few clusters, but more code. The percentile method is the basic method of the plan's reference.
- **numpy:** a new dependency for four small functions. Drawing 10,000 resamples of 10 scenarios takes about 6 ms in pure Python.

## Risks / Trade-offs

- **[Summed call times exceed wall time when tool calls run in parallel]** → They are documented as time spent in calls. Wall time remains the elapsed measure.
- **[Model time is measured on the client, so it includes HTTP, queueing and the OpenAI client's retries]** → That is the time the agent waits for. Only one run uses the server at a time, so queueing is small.
- **[Tool calls that vLLM's parser did not recognise are not counted as malformed]** → They are model-specific text. The MLflow traces keep the raw replies for the failure analysis.
- **[With 10 scenarios, the percentile bootstrap is slightly anti-conservative]** → The report shows the number of scenarios per comparison. Borderline intervals belong in the paper's threats to validity.
- **[The Wilson interval ignores clustering]** → It is descriptive. Claims about levels use the cluster bootstrap.
- **[Old and new records in one file]** → The means of new measurements skip unknown values. The wall time of old records still includes scoring, but wall time is indicative only.
- **[File order is no longer level order]** → Readers must not rely on file order. The report and CSV sort by model and level.

## Migration Plan

1. No configuration hash changes: existing results files resume and report as before, and new records carry the new fields.
2. Re-render existing reports. Success changes only for `step_limit` and `failed` runs that passed every check; the pilot has none.
3. Update `methods.tex` afterwards (outside the repo change): malformed calls are counted, and wall time excludes scoring.
4. Rollback: revert the commit. Records written in the meantime keep loading, because older code ignores unknown fields.

## Open Questions

- Should H1a also get an interval for the difference in tokens per success? A resample without any success makes that ratio undefined. `cluster_bootstrap` can add it later without changing this design.
