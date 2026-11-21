# Design

## Context

See proposal.md for the motivation and the delta specs for the required behaviour. Current state that shapes the approach:

- **Callback seam.** `runner.run_single` passes `callbacks=[meter, steps]` (`RunMeter`, `StepLogger`) to `run_task`. Both see every model call of the agent and of tools and every tool call. `StepLogger` already detects both loop signals, but only logs them; LangChain catches anything a handler raises unless the handler sets `raise_error = True`.
- **Where a run can be stopped.** The agent is a `create_agent` graph: a model node, then `ToolNode`, which runs parallel calls in a thread pool and wraps each call in `ToolErrorMiddleware`. An exception raised in the model node leaves `agent.invoke` unchanged; one raised during a tool call may be turned into a tool message by the middleware, or raised in a worker thread.
- **Tool errors.** A tool that raises fires `on_tool_error` before the middleware turns the error into a tool message, so `RunMeter` counts it. A tool that returns normally fires `on_tool_end`.
- **Tool-internal model calls.** Level-3 tools call the model about 30 times per review. Those calls happen while a tool run is active.
- **Status handling.** `run_single` maps `GraphRecursionError` to `step_limit` and any other exception to `failed` or `error`. `RunRecord.success` is already `status == "completed" and every check passes`, so a new status never succeeds.
- **Fingerprint.** `config_fingerprint` hashes every setting that changes results; resuming a file with another hash is refused.

## Goals / Non-Goals

**Goals:**
- Stop the two observed loop patterns at the first point where they are certain, before spending more model calls.
- Record those runs distinctly (`loop`) so the report can tell them from the step limit.

**Non-Goals:**
- **A wall-clock limit per run.** It depends on the GPU, so the same configuration would stop different runs on the Mac and the RTX.
- **Lowering `AGENT_MAX_STEPS`.** One-tool-per-turn models need about 33 calls for a good level-1 run.
- **Detecting cycles of several different calls** (A, B, A, B, ...). The step limit still covers them.
- **Changing the `run-logging` warnings.**

## Decisions

### A separate handler that raises, next to the observing `StepLogger`
A new `LoopGuard` callback handler (`evaluation/loop_guard.py`) with `raise_error = True`, passed last in `[meter, steps, guard]` so the meter and the step log record the reply that triggers it. `StepLogger` keeps its promise that logging never changes a run.
- *Alternative:* make `StepLogger` raise. Rejected: it mixes observing and stopping, and the spec says a logging failure never stops a run.
- *Alternative:* a `create_agent` middleware (`after_model` hook) that ends the graph. Rejected for now: the guard needs per-run state across the model and the tool node, and callbacks already deliver both with per-run instances.

### Agent calls are those made while no tool is running
The guard counts the tool runs that have started and not ended. A model event while that count is zero is a reply of the agent; any other is a tool-internal call and is ignored. This holds for nested tool chains, where the model call's direct parent is not the tool run (the parent-run test that `StepLogger` uses would miss those).

### The repeated-call check runs on the model's reply, before any tool starts
In `on_llm_end` of an agent reply, the guard walks the reply's parsed `tool_calls` in order and keeps the key `json.dumps([name, args], sort_keys=True)` of the last call and how many times in a row it was requested. When the count reaches the limit, it raises. Stopping here means the repeated call never runs (in the level-3 loop each call redid a whole review), and the exception leaves the model node in the main thread, not a `ToolNode` worker where the middleware could swallow it.
- Arguments are compared as the model wrote them, before the JSON-string middleware decodes them. Two calls that differ only in that encoding count as different; that only delays the stop.
- Malformed calls (in `invalid_tool_calls`) are not counted; they never run.

### The error-streak check runs before the next agent reply
`on_tool_error` increments the streak and `on_tool_end` resets it, under a lock (parallel calls end in worker threads). In `on_chat_model_start` of an agent reply, the guard raises when the streak has reached the limit. The model is then not called again.

### `AgentLoopError` is not a `ToolEvalError`
`ToolEvalError` means "report back to the agent". The new error derives from `Exception` directly, carries the signal and a short description of the call or tool, and `run_single` catches it before the generic handler: status `loop`, `error` = its message, one WARNING line. Scoring then runs as for `step_limit`, so a loop run keeps partial credit.
If LangChain or LangGraph wraps the exception, the runner unwraps it via `__cause__`/`__context__`; a test with the real `create_agent` loop pins down which applies.

### Settings and record
- `Settings.agent_max_repeated_calls: int = Field(default=8, ge=0)` and `agent_max_tool_error_streak: int = Field(default=10, ge=0)`, from `AGENT_MAX_REPEATED_CALLS` and `AGENT_MAX_TOOL_ERROR_STREAK`; 0 switches a check off.
- Both enter `config_fingerprint`, the MLflow parameters, and `RunRecord` as `max_repeated_calls: int | None = None` and `max_tool_error_streak: int | None = None` (None for older records).
- `RunStatus` gains `"loop"`; `LevelSummary` gains `loop_runs`, and the quality table a `loop` column after `step limit`.

### Defaults: 8 identical calls, 10 failed calls
- **8 identical calls in a row.** The first draft used 5. In the RTX screen on 2026-11-21, qwen at level 2 (`level2-protein-2022h2-01`) called `write_review({})` four times in a row (the long `markdown` argument was lost, so validation failed each time) and then finished with a success; at level 3 it repeated `conduct_literature_review` three times in a row before finishing. A limit of 5 left a margin of one call, so the default is 8. In the observed qwen-weak loop, stopping at 8 instead of 100 still saves about 92 reviews.
- **10 failed calls in a row.** Searches with no result are tool errors, and a weak model may legitimately try several queries before one works; 10 leaves room for that. The observed streak was 99.

## Risks / Trade-offs

- [A sound run is stopped] → Check the 90 pilot runs' MLflow traces on the RTX before the full experiment: no `completed` run may contain 8 identical calls in a row or 10 failed calls in a row. If one does, raise the default before the full run, not after.
- [Cost metrics of weak models drop] → Loop runs now stop early, so tokens per run and wall time of qwen-weak fall. Tokens per success are unaffected (loop runs never succeed). The paper must say that loop runs are cut short; `methods.tex` gets a sentence.
- [Old results cannot be resumed] → Intended: the hash changes. The pilot files move to `results/old/` before the full run; `report` still reads them.
- [The exception is swallowed by a LangChain layer] → Covered by an end-to-end test with the real `create_agent` loop and a fake model, for both signals.

## Migration Plan

1. Implement and test locally; push the branch.
2. On the RTX: pull, move `results/presets-*.jsonl` to `results/old/pilot-1116/`, check the pilot traces against the defaults.
3. Run `make eval-presets` with `REPEATS=1` for all presets as a screen, then `REPEATS=10`.

Rollback: set both limits to 0. That restores the old behaviour but not the old hash, since the limits are hashed even when off.
