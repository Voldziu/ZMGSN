# Proposal

## Why

Some runs are lost long before the step limit ends them, and the full experiment (five models, about 3,000 runs) pays for every one of them in GPU time. On 2026-11-19 qwen-weak never ended a level-3 run. In one repeat it invented a new query on each call and got 99 tool errors in a row; in another it re-ran the same successful `conduct_literature_review` call about 50 times, and each call redid the whole review (about 30 tool-internal model calls). Both runs could only end at the 100-step limit, after up to about 70 minutes on the Mac. The `run-logging` change already warns about both patterns (the same call three times in a row, five failed calls in a row), but the warnings only observe. The group decided on 2026-11-21 to stop such runs instead of waiting for the step limit.

## What Changes

- **Loop guard:** the agent's run stops as soon as it is clearly stuck, on either of two signals counted on the agent's own tool calls (tool-internal model calls do not count):
  - **repeated call:** the agent requests the same tool call (same tool, same arguments) for the `AGENT_MAX_REPEATED_CALLS`-th time in a row, default **8**. The run stops before that call executes.
  - **error streak:** `AGENT_MAX_TOOL_ERROR_STREAK` tool calls in a row fail, default **10**. The run stops before the next reply of the agent's model.
  - `0` switches a signal off.
- **New run status `loop`:** a stopped run is recorded with status `loop` and an error message naming the signal and the call. Like `step_limit`, it is scored (partial credit) but never a success.
- **BREAKING for resuming old results:** both limits enter the configuration hash, and the run record and the MLflow parameters carry them. Results files written before this change cannot be resumed with it; the pilot files must be moved aside before the full experiment.
- **Report:** the quality table gets a `loop` column next to `step limit`.
- The `run-logging` warnings stay as they are (third identical call, fifth failed call); they fire before the guard.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `agent-tool-calling`: a new requirement that stuck runs are stopped early and recorded as `loop`.
- `review-scoring`: `loop` runs never succeed but keep partial credit, like `step_limit` runs.
- `llm-client-config`: the configuration hash and the run record cover both loop-guard limits.
- `run-tracing`: the MLflow parameters include both limits.
- `run-logging`: the warnings still only observe, but a run they warn about may now be ended by the loop guard, not only by the step limit.
- `evaluation-report`: the quality table counts `loop` runs.

## Impact

- **Code:** `src/tool_eval/config.py` (two settings), a new callback handler in `src/tool_eval/evaluation/` passed by the runner next to `RunMeter` and `StepLogger`, `src/tool_eval/exceptions.py` (an error that is not reported back to the agent), `src/tool_eval/evaluation/runner.py` (status mapping, fingerprint, record and MLflow parameters), `src/tool_eval/evaluation/models.py` (status and two record fields), `src/tool_eval/evaluation/report.py`.
- **Protocol:** loop runs stop earlier, so their tokens and wall time are lower than they would be at the step limit. Cost metrics of weak models drop; the success rate does not change, since those runs failed anyway. `methods.tex` must describe the guard.
- **Results:** a new configuration hash; old results files are not resumable (they stay readable by `report`).
- **Dependencies:** none.
