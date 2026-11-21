# Proposal

## Why

The full experiment runs five models one after another for many hours, and the console only shows one log line per run. It does not show which model of the five is running, how far the current model is, or how long it will take. `eval-presets.sh` starts one Python process per model, so no process knows the whole plan by itself.

## What Changes

- **Progress bars on the console** while `run` executes, four nested bars:
  - **model:** position of the current preset among the presets of `eval-presets.sh` (for example `2/5 gemma`);
  - **runs:** runs of this model, finished out of planned, with an ETA; runs skipped because they already finished count as done;
  - **repeat:** the current repeat out of the requested repeats;
  - **scenario:** the current scenario of the repeat, with the running level and its position in the block.
- **Only on a terminal:** the bars are drawn only when stderr is a terminal. With stderr redirected to a file or pipe (for example `| tee`), nothing changes.
- **Log lines stay readable:** console log lines are printed above the bars instead of breaking them. The log file next to the results is unchanged.
- **`eval-presets.sh`** tells each run its preset position through `TOOL_EVAL_PRESET_POSITION` (`<index>/<count>`). Without it, the model bar shows `1/1`.

## Capabilities

### New Capabilities

- `run-progress`: what the console shows about an evaluation's progress and when it shows it.

### Modified Capabilities

None. The log file, run records and configuration hash are unchanged.

## Impact

- **Code:** a new module `src/tool_eval/evaluation/progress.py`; `runner.run_evaluation` and `_execute_runs` drive it; `utils/logger.py` writes console lines through tqdm; `server/eval-presets.sh` sets the preset position.
- **Dependencies:** `tqdm` (runtime), `types-tqdm` (dev).
- **Results and protocol:** unchanged.
