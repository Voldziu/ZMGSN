# Tasks

## 1. Logging set-up and the log file

- [x] 1.1 In `utils/logger.py`, remove the set-up at import, `LOG_FILE` and `TOOL_EVAL_LOG_FILE`, and add `configure_logging(log_file)`: a console handler at `TOOL_EVAL_LOG_LEVEL` (default INFO) and, when a file is given, an appending DEBUG file handler without rotation, both on the root logger (root at WARNING, `tool_eval` at DEBUG). A repeated call closes and replaces the handlers installed before. Verify with tests:
  - importing `tool_eval` modules and calling `get_logger` installs no handler;
  - after `configure_logging(file)`, a `tool_eval` DEBUG line reaches the file but not the console, and an INFO line reaches both;
  - with `TOOL_EVAL_LOG_LEVEL=DEBUG`, DEBUG lines reach the console;
  - a second call closes the first file and writes only to the new one;
  - a WARNING from a library logger such as `langchain_core` reaches the file.
- [x] 1.2 In `main.py`, make `run` call `configure_logging` with its `--output` path and the suffix `.log`, and reject an `--output` whose suffix is `.log` with exit code 2 before any run starts. Make `report` call it without a file. Verify with command-line tests in `tmp_path` and a fake agent:
  - `run --output <tmp>/presets-x.jsonl` writes `<tmp>/presets-x.log`;
  - a second invocation with the same output keeps the first invocation's lines and appends after them;
  - `report` writes no log file;
  - `run --output <tmp>/x.log` exits with code 2 and executes no run.
- [x] 1.3 In the README, state that each evaluation's log is next to its results file, that the console shows INFO and above while the file keeps DEBUG, and that `TOOL_EVAL_LOG_LEVEL=DEBUG` shows the steps live. Verify that the README matches the code.

## 2. Run ID and run lines

- [x] 2.1 In `utils/logger.py`, add the run context (a `ContextVar` and `run_context(run_id)`) and a filter that sets `run_tag`. Put the filter and the format `%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(run_tag)s%(message)s` on both handlers. Verify with tests:
  - a line logged inside `run_context("r1")` shows `[r1] ` before its message, and a line outside shows no tag;
  - in a `create_agent` loop driven by a fake model inside `run_context`, a warning that a tool logs through `tool_eval.services.arxiv`'s logger carries the run ID.
- [x] 2.2 In `run_single`, enter `run_context(run_id)` around the whole run, from opening the MLflow run to logging its results. Verify with runner tests that `Run <id> failed` and an MLflow warning (a fake MLflow whose `log_metrics` fails) carry the run ID.
- [x] 2.3 Make `run_evaluation` write the evaluation start line (model, levels, scenario IDs, repeats, order seed, finished runs, runs still to execute) instead of `Level order seed`. Verify with a runner test that resumes a file holding four finished runs of three levels, one scenario and two repeats: the line names the order seed and says four runs are finished and two are still to execute.
- [x] 2.4 Make `run_single` write `start: position p/n, seed=<seed>` before building the agent, with `n` passed by `_execute_runs` as the length of the block's plan. Verify with runner tests:
  - every run's start line precedes its other lines;
  - with three levels in the plan, the run at position 2 says `position 2/3`;
  - a run whose agent raises still has its start line.
- [x] 2.5 Move the run end line from `_execute_runs` into `run_single`, inside the run context, and add wall time, tool calls and tool errors. Verify with a runner test that the end line of a `step_limit` run shows its status, `success=False`, and the tokens, wall time, tool calls and tool errors of its record.
- [x] 2.6 In the README, describe the evaluation start, run start and run end lines and the run ID on every line written during a run. Verify that the README matches the code.

## 3. Step lines and loop warnings

- [x] 3.1 Add a module in `src/tool_eval/evaluation/` with `StepLogger(BaseCallbackHandler)`. It writes a DEBUG line for each agent reply (step number, tool calls, malformed calls, a finish reason other than `stop` or `tool_calls`, duration, start of the text) and for each tool call (name, arguments, result or error type and message, duration, tool-internal model calls). Arguments, results and text are cut to 150 characters on one line. Verify with unit tests on fake callback events, as in `test_metering.py`:
  - a reply with one unparseable call and one call to a name the level does not offer counts two malformed calls;
  - `finish_reason="length"` shows `finish=length`, while `stop` and `tool_calls` show no finish reason;
  - a failed tool call shows the tool name, its arguments, the error type and its message;
  - a 5,000-character argument is cut to 150 characters ending in `...`, and no line contains a line break;
  - model calls whose parent run is a tool run write no line, and the tool's line shows their number and summed duration (fake clock).
- [x] 3.2 Add the loop warnings to `StepLogger`: the third identical tool call in a row (key: tool name and arguments as JSON with sorted keys), and five failed tool calls in a row. Each warning is written at most once per run. Verify with unit tests:
  - three identical calls give one warning, and ten identical calls still give one;
  - three calls with different arguments give none;
  - two calls whose arguments share the first 150 characters but differ later are not identical;
  - five failures in a row with different arguments give one warning naming the tool;
  - four failures, one success and four more failures give none.
- [x] 3.3 Pass `StepLogger` to `run_task` in `run_single` next to `RunMeter`, with the level's tool names. Verify with an agent-loop test (fake model, real `create_agent`, log file in `tmp_path`):
  - the reply and tool lines appear in order, between the run's start and end lines;
  - a run stopped by the step limit keeps its step lines;
  - an exception raised inside the handler (patched) leaves the run's status and record unchanged.
- [x] 3.4 In the README, describe the step lines and the loop warnings, and state that the warnings only observe and never stop a run. Verify that the README matches the code.

## 4. Integration checks

- [x] 4.1 Run `make test`, `uv run ruff format --check .` and `openspec validate run-logging --strict`. List the files under `logs/` and `results/` with their modification times before and after `make test`, and verify that the test suite changed none of them.
- [x] 4.2 Verify that the configuration hash is unchanged: with `LLM_MODEL=qwen-weak`, resume a copy of the 2026-11-19 results file `results/presets-qwen-weak.jsonl` (or another results file written before this change) with `--only rag-2024h1 --levels level2 --repeats 2`. It executes no run and reports no configuration error.
- [ ] 4.3 On the Mac, with `make mac-serve PRESET=qwen-weak` running, run `LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=qwen-weak uv run python -m tool_eval.main run --only rag-2024h1 --levels level3 --repeats 1 --output results/check-run-logging.jsonl --runs-dir workspace/check-runs`. Verify:
  - `results/check-run-logging.log` holds the evaluation start line, the run's start line, its step lines, a loop warning before the run ends, and the end line;
  - the console shows the loop warning but no step lines.

## Workflow follow-up

- Archive the change with `/opsx:archive` once it is implemented and checked.
