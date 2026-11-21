# Tasks

## 1. Progress bars

- [x] 1.1 Add `tqdm` and `types-tqdm`, and implement `Progress` in `evaluation/progress.py` (four bars, hooks, preset position from `TOOL_EVAL_PRESET_POSITION`). Verify with tests on a `StringIO`: the output names `2/5 qwen` when the variable is `2/5`, `1/1` without it, the scenario id and level, and a disabled `Progress` writes nothing.
- [x] 1.2 Drive `Progress` from `run_evaluation` and `_execute_runs`, with `progress_file` defaulting to a terminal stderr. Verify with a runner test that resumes a file with finished runs: the runs bar starts at the finished count and ends at the planned count, and with a non-terminal stderr no bar output appears.
- [x] 1.3 Route console log lines through `tqdm.write` in `utils/logger.py`. Verify that existing logging tests pass unchanged and that a console line still carries the run ID.
- [x] 1.4 Export `TOOL_EVAL_PRESET_POSITION` per preset in `eval-presets.sh`, and describe the bars in the README. Verify with `bash -n` and that the README matches the code.

## 2. Integration

- [x] 2.1 Run `make test` and verify that it passes.
- [ ] 2.2 Run `make eval-presets PRESETS=qwen-mid REPEATS=1 ONLY="..." LEVELS="level1 level2 level3"` on the Mac in a terminal and verify that the four bars show and update.
