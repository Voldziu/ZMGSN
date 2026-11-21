# Design

## Context

See proposal.md for the motivation and the `run-logging` spec for the required behaviour. Current state that shapes the approach:

- **Set-up at import.** `utils/logger.py` installs a rotating file handler (`logs/tool_eval.log`, relative to the working directory, 5 MB × 3) and a console handler on the `tool_eval` logger the first time a module calls `get_logger`. Every module calls it at import, so importing the package is enough to write to the file. The `tool_eval` logger does not propagate.
- **What is logged today.**
  - service warnings from `services/arxiv.py`, `pdf.py`, `wikipedia.py` and `reports.py`;
  - MLflow warnings from `evaluation/tracing.py`;
  - `Run <id> failed` with its traceback, the order seed, and one line per finished run from the runner;
  - the command line's error and summary lines from `main.py`.
- **Callback seam.** `runner.run_single` passes `callbacks=[meter]` to `run_task`. `RunMeter` already receives every model call of the agent and of tools (`on_chat_model_start`, `on_llm_end`) and every tool call (`on_tool_start`, `on_tool_end`, `on_tool_error`). It attributes a model call to a tool when the call's parent run is a tool run. Parallel tool calls end in `ToolNode`'s thread pool, so the meter takes a lock.
- **Tool events.** `on_tool_start` receives the tool name (`serialized["name"]`) and the arguments the tool gets (`inputs`), after the JSON-string middleware has decoded them.
- **Lost messages.** On `step_limit` runs and runs that raise, `run_task` raises and the message history is lost. Anything about the steps has to be written while they happen.
- **Per-run context reaches tools.** `use_settings` puts each run's settings into a `ContextVar` (`config.py`), and tools read their workspace from it, also inside `ToolNode`'s threads.
- **Callback failures.** LangChain catches an exception raised in a callback handler, logs `Error in <handler>.<event> callback` as a warning through its own logger and continues, unless the handler sets `raise_error`.
- **Callers of the logger.** `main.py` (`run` and `report`), the tests (through `run_evaluation` and through `main`), LangGraph Studio (`studio.py`) and ad-hoc scripts.

## Goals / Non-Goals

**Goals:**
- One log per evaluation that shows each run's steps and flags loops while the run is still going, without MLflow.
- No effect on what is measured: the same records, the same configuration hash, and negligible extra time.

**Non-Goals:**
- **Full content locally:** complete messages, arguments and tool results stay in MLflow traces. There are no local transcripts.
- **Stopping loops:** a loop guard or a time limit would change the protocol and the configuration hash, so it is a group decision.
- **Loop statistics in run records:** the warnings are not counted in `CostMetrics`.
- **Library logs below WARNING:** httpx, LangChain and MLflow stay at WARNING.
- **Structured logs:** lines stay plain text.

## Decisions

### The command line sets up logging, and importing sets up nothing
- **Set-up function.** `utils/logger.py` gets `configure_logging(log_file: Path | None)`:
  - a console handler on stderr, at the level from `TOOL_EVAL_LOG_LEVEL` (default INFO);
  - when `log_file` is given, a file handler at DEBUG that appends (`mode="a"`, UTF-8, no rotation), after creating the file's directory;
  - the `tool_eval` logger at DEBUG.
- **Root logger.** The handlers go on the root logger, which stays at WARNING. `tool_eval` lines of every level and warnings from libraries that propagate reach both handlers, so LangChain's report of a failing callback lands in the file too.
- **Repeated calls.** The function removes and closes the handlers it installed before, so tests can call `main` several times in one process.
- **`get_logger`.** It keeps its name and returns `logging.getLogger(name)` without installing handlers. `TOOL_EVAL_LOG_FILE` and `LOG_FILE` go away.
- **Callers.**
  - `main run` calls it with `args.output.with_suffix(".log")`;
  - `main report` calls it without a file;
  - Studio and scripts call nothing and get Python's defaults, which send warnings to stderr.
- **Tests.** Code that calls `run_evaluation` directly configures nothing, so its records propagate to the root logger, where pytest captures them. Tests that call `main` pass an output in `tmp_path` or change into it, so their log lands there.

*Alternatives:*
- **Keep the set-up at import and redirect it in `tests/conftest.py`:** the smallest change. But the log stays relative to the working directory, and one shared file mixes the models, whose run IDs are identical.
- **Handlers on the `tool_eval` logger without propagation, as today:** library warnings, including failing callbacks, would bypass the file.

### Steps are written by a second callback handler
- **Where.** A new module in `src/tool_eval/evaluation/` holds `StepLogger(BaseCallbackHandler)`. `run_single` passes `callbacks=[meter, step_logger]` and gives it the level's tool names, as it does for the meter. It writes through its own `tool_eval.evaluation.*` logger.
- **Why a callback.** It sees what `RunMeter` sees, including the model calls inside tools that agent middleware cannot see. It writes each line when its event ends, so `step_limit` and killed runs keep their steps.
- **Agent replies.** In `on_llm_end` for an agent-side call (the parent run is not a tool run, as in the meter), it increments the step number and writes one DEBUG line:
  - `#n model -> k tool calls`;
  - malformed calls: `invalid_tool_calls`, and calls to names the level does not offer;
  - `finish=<reason>` when the reason is neither `stop` nor `tool_calls`;
  - the duration and the start of the text.
- **Tool calls.** `on_tool_start` stores the tool name, its arguments and the start time. `on_tool_end` and `on_tool_error` write one DEBUG line: `#n name(args) -> ok: <result>` or `#n name(args) -> error: <Type>: <message>`, with the duration and the tool-internal model calls. The step number is that of the latest agent reply, which requested the call.
- **Model calls inside a tool.** A model call whose parent run is a tool run adds to that tool run's count and summed duration and writes no line. Like the meter, the handler checks only the direct parent.
- **Shortening.** Arguments are rendered as `key=value` pairs with `repr` values. Results are the `ToolMessage` content. Runs of whitespace, including line breaks, collapse to one space, and each part is cut to 150 characters, ending in `...`.
- **Threads.** A lock guards the handler's state, as in `RunMeter`.
- **Failures.** `raise_error` stays `False`, so an exception inside the handler is logged by LangChain and the run continues.

*Alternatives:*
- **Extend `RunMeter`:** it would mix measuring with logging, and the meter's tests pin its counters. A separate handler can be tested on its own.
- **`wrap_model_call` and `wrap_tool_call` middleware:** they do not see the model calls made inside tools.
- **Write the message history when the run ends:** it is lost on exactly the runs that need it.

### The step handler watches for loops
- **Repeated call.** The handler compares each agent tool call, in start order, with the calls before it. The key is the tool name and the full arguments as JSON with sorted keys. When the same key starts for the third time in a row, it writes one WARNING with the shortened call and never writes this warning again in that run.
- **Error streak.** It counts the tool calls that ended in `on_tool_error` since the last one that ended in `on_tool_end`. At five it writes one WARNING that names the tool, once per run.
- **Full arguments.** The comparison uses the full arguments, so two calls whose shortened text looks the same but whose arguments differ are not identical.
- **Constants.** The thresholds 3 and 5 are module constants, not settings, so they never enter the configuration hash.
- **Why two signals.** In the 2026-11-19 level-3 runs, one repeat invented a new query for every call and each call failed, so no two calls were identical. The other repeated the same successful call about 50 times, so no call failed.

### A context variable puts the run ID on every line
- **Context.** `utils/logger.py` holds a `ContextVar` with the current run ID and a context manager `run_context(run_id)`. `run_single` enters it around the whole run, from opening the MLflow run to logging the run's results, so MLflow warnings about the run carry the ID too.
- **Filter.** A `logging.Filter` on both handlers adds `record.run_tag`: `[<run_id>] ` inside a run, empty outside. The format becomes `%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(run_tag)s%(message)s`.
- **Threads.** The variable reaches tools in `ToolNode`'s threads in the same way as the run's settings.

*Alternative:* pass the run ID to every service call. Rejected: it would change service signatures only for logging.

### The runner writes the evaluation, run start and run end lines
- **Evaluation start.** `run_evaluation` writes one INFO line with the model, the levels, the scenario IDs, the repeats, the order seed, the number of finished runs in the results file and the number of planned runs of the requested levels that are not finished. It replaces the `Level order seed` line.
- **Run start.** `run_single` writes `start: position p/n, seed=<seed>` at INFO inside the run context, after drawing the seed and before building the agent. `_execute_runs` passes the length of the block's plan as `n`.
- **Run end.** The end line moves from `_execute_runs` into `run_single`, after the results are logged to MLflow and still inside the run context. It adds wall time, tool calls and tool errors to status, success and tokens.

### One file per results file, without rotation
- **One model per file.** The configuration hash includes the model, so a results file holds one model, and so does its log.
- **No rotation.** At about 600 runs per model and a few dozen short lines per run, a file stays at a few megabytes. Rotation would drop the start of a long evaluation.
- **Flushing.** The file handler flushes every record, so a killed process loses at most the line being written.

## Risks / Trade-offs

- [A level-1 agent may legitimately read the same file three times in a row] → The warning only observes. The threshold is a constant that can change without touching the protocol.
- [Loops that alternate between two calls (A, B, A, B) are not flagged as repeated calls] → The error streak still catches alternations that fail. A count over the whole run can replace "in a row" if such loops show up.
- [The handler's work counts towards `wall_seconds`] → It formats a few short strings per event, against model calls that take seconds.
- [Studio and scripts lose the file log] → Their warnings still reach stderr, and MLflow tracing records everything when it is on.
- [An `--output` that ends in `.log` would make the log and the results the same file] → `run` rejects such an output path before any run starts.
- [The line format changes] → Nothing in the repository parses the old log.

## Migration Plan

- No data migration. Existing `logs/tool_eval.log` files stay where they are and are no longer written.
- Rollback: revert the change. Results files are not affected either way.
