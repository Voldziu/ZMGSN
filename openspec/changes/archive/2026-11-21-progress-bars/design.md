# Design

## Context

See proposal.md. `run_evaluation` plans runs by repeat, then scenario, then a random level order per block (`block_order`), and `_execute_runs` walks that plan, skipping finished runs. Each preset of `eval-presets.sh` is a separate `uv run python -m tool_eval.main run` process. The console handler from `configure_logging` writes to stderr.

## Decisions

### A `Progress` object owned by `run_evaluation`
`evaluation/progress.py` holds `Progress`, a context manager with four `tqdm` bars at positions 0–3 and the hooks `repeat(n)`, `scenario(index, id)`, `run(level, position, size)` and `run_done()`. `run_evaluation` creates it after counting planned and finished runs; `_execute_runs` calls the hooks. When disabled, every hook does nothing.
- *Why not in `main.py`:* the plan (planned and finished runs) is known only inside `run_evaluation`.

### Enabled only on a terminal
`run_evaluation(..., progress_file=None)`: `None` means `sys.stderr` if it is a terminal, otherwise no bars. Tests pass a `StringIO` to force bars on.

### Console lines go through `tqdm.write`
The console handler gets a small stream object whose `write` calls `tqdm.write(text, file=sys.stderr, end="")`. With no bars open, `tqdm.write` just writes the text, so output is the same as before. `tqdm.contrib.logging.logging_redirect_tqdm` was rejected: it replaces handlers and drops their filters, which would lose the run-ID filter.

### Preset position from the environment
`eval-presets.sh` exports `TOOL_EVAL_PRESET_POSITION=<i>/<n>` for each preset. `Progress` parses it; a missing or malformed value gives `1/1`.

## Risks / Trade-offs

- [Bars in a file] → Only drawn on a terminal; the screen and full runs on the RTX are piped through `tee`, so their output files stay clean. To see bars there, attach to the tmux session without `tee`.
- [Inner bars flicker on fast runs] → Runs take tens of seconds, so redraws are rare.
