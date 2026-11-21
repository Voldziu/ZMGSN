# Tasks

## 1. Settings, status and record

- [x] 1.1 Add `agent_max_repeated_calls` (default 8, ≥ 0) and `agent_max_tool_error_streak` (default 10, ≥ 0) to `Settings`. Verify with tests that the defaults are 8 and 10, that `AGENT_MAX_REPEATED_CALLS=0` is accepted and that a negative value is rejected.
- [x] 1.2 Add both limits to `config_fingerprint`, the run's MLflow parameters and `RunRecord` (`max_repeated_calls`, `max_tool_error_streak`, both `int | None = None`), and add `"loop"` to `RunStatus`. Verify with tests that changing either limit changes the hash, that a new record stores 8 and 10, and that a record line without the fields still loads with both unknown.
- [x] 1.3 Add `AgentLoopError(Exception)` to `exceptions.py`, outside the `ToolEvalError` hierarchy. Verify with a test that it is not a `ToolEvalError`.

## 2. The guard

- [x] 2.1 Implement `LoopGuard` in `evaluation/loop_guard.py` (callback handler, `raise_error = True`, lock, active-tool count): the repeated-call check in `on_llm_end` of agent replies and the error-streak check in `on_chat_model_start` of agent replies. Verify with unit tests that drive the callbacks directly:
  - the fifth identical requested call raises, naming the tool and its arguments; four do not;
  - a different call in between resets the count; different arguments never raise;
  - five identical calls in one reply raise;
  - ten failed tool calls followed by an agent reply raise before the reply; nine, a success and nine more do not;
  - model calls made while a tool run is active count for neither check;
  - a limit of 0 switches its check off.
- [x] 2.2 Pass the guard last in the runner's callbacks and map `AgentLoopError` to status `loop` with its message as `error`, a WARNING line, and scoring as for `step_limit`. Verify with runner tests through the real `create_agent` loop and a fake model:
  - a fake that repeats one `read_file` call forever ends as `loop` after 7 executed calls, with the end line `end: loop, success=False, ...` and the warning in the log;
  - a fake whose tool calls all fail ends as `loop` after 10 tool errors, and the fake is not called an eleventh time;
  - a fake that finishes normally is unaffected, and one that alternates two calls ends as `step_limit` with `agent_max_steps=6`;
  - a `loop` run whose review passes every check has score 1.0 and no success.

## 3. Report and docs

- [x] 3.1 Add `loop_runs` to `LevelSummary` and a `loop` column after `step limit` in the quality table (and the CSV export if it carries `step_limit_runs`). Verify with report tests: 7 loop runs and 1 step-limit run in one group show 7 and 1; an older file shows 0.
- [x] 3.2 Document the guard in the README (both settings, the `loop` status, that old results files are not resumable) and add one sentence on it to `docs/overleaf` `methods.tex` where the step limit is described. Verify that the README and the paper match the code.

## 4. Integration

- [x] 4.1 Run `make test` (pytest, ruff, mypy) and verify that it passes.
- [x] 4.2 On the RTX, query the pilot runs' MLflow traces and verify that no `completed` run has 8 identical tool calls in a row or 10 failed tool calls in a row; if one does, raise the default and record why in design.md.
- [x] 4.3 On the RTX, after moving the pilot results to `results/old/`, run `make eval-presets PRESETS=qwen-weak REPEATS=1 ONLY= LEVELS="level1 level2 level3"` and verify that level-3 loops end as `loop` well before the step limit.

## Workflow follow-up

- Archive the change after the RTX checks.

Notes on 4.2 and 4.3 (2026-11-21): the MLflow trace scan was superseded by the RTX screen (150 runs, 5 models, run with the guard on at `max_repeated_calls=5`, `max_tool_error_streak=10`). Every `completed` run stayed below 5 identical calls in a row and below 10 failed calls in a row, or the guard would have stopped it, so no completed run reaches the defaults of 8 and 10. qwen-weak level 3 ended as `loop` in 9 of 10 runs, at about 70 s each instead of the step limit. On the Mac, the guard stopped qwen-mid `level3-xai-2022-01` at the 8th identical call.
