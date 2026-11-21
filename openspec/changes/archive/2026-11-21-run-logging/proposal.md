# Proposal

## Why

The evaluation log cannot show what an agent did, and most of what it holds comes from tests. On 2026-11-19 a local qwen-weak run spent 8 minutes in a level-3 loop: 100 tool calls, 99 tool errors and 557,437 tokens. Its log got nine arXiv rate-limit warnings and one final `step_limit` line. A second level-3 repeat, which re-ran the same review about 50 times, was stopped before it finished and left no record and no log line. Finding out what had happened took an ad-hoc replay script. The full experiment (about 3,000 runs, five models) needs a log that shows each run's steps and flags loops while they happen, also when MLflow tracing is off.

- **No agent steps.** `logs/tool_eval.log` gets service warnings (arXiv, PDF, Wikipedia), errors and one line per finished run. The agent's replies, the arguments it passes to tools and what the tools return appear only in MLflow traces, which are opt-in and need a server.
- **Stopped runs leave nothing.** The only line about a run is written when it ends. On `step_limit` runs and runs that end with an exception, `run_task` raises and the message history is lost, so nothing can be written afterwards either.
- **Test noise.** pytest writes into the same file. In the logs of the two checkouts, 1,410 of 1,414 and 822 of 830 run lines come from test scenarios.
- **The log depends on the working directory.** The default path `logs/tool_eval.log` is relative, so every checkout and worktree keeps its own log. The 2026-11-19 run logged into a worktree, not into the main checkout.
- **Lines do not name their run.** Service warnings carry no run ID, and run IDs such as `level3-rag-2024h1-01` do not name the model, so one shared log cannot tell the models' runs apart.

## What Changes

- **Log next to the results file:** `run --output results/presets-qwen.jsonl` writes its log to `results/presets-qwen.log` and appends to it when the evaluation is resumed. Only the `run` command writes a log file; importing `tool_eval` installs no handlers.
- **BREAKING for readers of the old log:** `logs/tool_eval.log` is no longer written, and `TOOL_EVAL_LOG_FILE` is removed. Neither was documented.
- **Run ID on every line:** every line written while a run executes carries the run ID, including warnings from services that tools call.
- **Start and end lines:** one line when an evaluation starts (model, levels, scenarios, repeats, order seed, runs done and runs to do) and one when each run starts. The line written when a run ends adds wall time, tool calls and tool errors.
- **One short line per agent step**, at DEBUG level, so only the file gets it by default:
  - each reply of the agent's model: how many tool calls it makes, malformed calls, a finish reason other than a normal stop, and the start of any text;
  - each tool call: the tool name, its shortened arguments, and its shortened result or its error, with the call's duration;
  - model calls made inside a tool: their number and time on the tool's line, not one line each.

  Arguments, results and text are cut to 150 characters and kept on one line. The full content stays in MLflow traces.
- **Loop warnings:** a WARNING when the agent makes the same tool call for the third time in a row, and when five tool calls in a row fail. Each warning fires at most once per run. The warnings only observe: the run continues, and no guard or time limit is added.
- **Levels:** the console shows INFO and above, as before (`TOOL_EVAL_LOG_LEVEL` changes it). The file keeps DEBUG and above and is not rotated.

## Capabilities

### New Capabilities

- `run-logging`: what the evaluation log records and where it is kept: one log per results file, the run ID on every line written during a run, start and end lines, one short line per agent step, loop warnings, and the guarantee that logging never changes runs or their results.

### Modified Capabilities

None. MLflow tracing (`run-tracing`) and the other capabilities keep their requirements.

## Impact

- **Code:**
  - `src/tool_eval/utils/logger.py`: no handlers at import; a set-up function for the command line; the run context and the filter that adds the run ID to lines;
  - `src/tool_eval/main.py`: `run` logs to the console and to the file next to `--output`; `report` logs to the console only;
  - `src/tool_eval/evaluation/runner.py`: the evaluation and run start lines, the run context, the longer end line, and a second callback handler next to `RunMeter`;
  - a new module in `src/tool_eval/evaluation/` for that handler.
- **Tests:** they no longer write to a shared log file. New tests cover the handler, the log file next to the results and the run ID on service warnings.
- **Docs:** README (where the log is, and how to see the steps live).
- **Results and protocol:** unchanged. The configuration hash, the run records and the agent's behaviour stay the same, so existing results files resume as before. The handler's work counts towards `wall_seconds`, but it is small next to the model calls.
- **Other entry points:** LangGraph Studio (`studio.py`) and ad-hoc scripts get no log file. Their warnings follow Python's defaults and go to stderr.
- **Old logs:** existing `logs/tool_eval.log` files stay where they are.
- **Dependencies:** none.
