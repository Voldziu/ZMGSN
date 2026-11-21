"""Console progress bars for an evaluation: model, runs, repeat and scenario.

`Progress` draws four nested `tqdm` bars while `run_evaluation` executes, but only when it is
given a stream (by default stderr, and only when stderr is a terminal). Disabled, every hook
does nothing. The bars never reach the log file or the run records.

`eval-presets.sh` runs one process per model and tells each its place in the preset list through
`TOOL_EVAL_PRESET_POSITION` (`<index>/<count>`); without it the model bar shows `1/1`.
"""

import os
import re
import sys
from types import TracebackType
from typing import Never, Self, TextIO

from tqdm import tqdm

PRESET_POSITION_VARIABLE = "TOOL_EVAL_PRESET_POSITION"
_POSITION = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def terminal_stderr() -> TextIO | None:
    """Return stderr when it is a terminal, else `None` (no bars)."""
    try:
        return sys.stderr if sys.stderr.isatty() else None
    except (AttributeError, ValueError):
        return None


def preset_position() -> tuple[int, int]:
    """The model's place among the presets of `eval-presets.sh`, `(1, 1)` when unknown."""
    match = _POSITION.match(os.environ.get(PRESET_POSITION_VARIABLE, ""))
    if match is None:
        return 1, 1
    index, count = int(match[1]), int(match[2])
    return (index, count) if 1 <= index <= count else (1, 1)


class Progress:
    """Four nested bars: model, runs of the model, repeat and scenario."""

    def __init__(
        self,
        *,
        model: str,
        planned_runs: int,
        finished_runs: int,
        repeats: int,
        scenarios: int,
        file: TextIO | None,
    ) -> None:
        """Prepare the bars; they open when the context is entered.

        Args:
            model: Served model name, shown on the model bar.
            planned_runs: Runs of the whole plan, including finished ones.
            finished_runs: Runs of the plan that already finished and will be skipped.
            repeats: Repeats of the plan.
            scenarios: Scenarios per repeat.
            file: Stream to draw on; `None` disables every bar.
        """
        self._file = file
        self._model = model
        self._planned = planned_runs
        self._finished = finished_runs
        self._repeats = repeats
        self._scenarios = scenarios
        self._bars: list[tqdm[Never]] = []

    @property
    def enabled(self) -> bool:
        """Whether the bars are drawn."""
        return self._file is not None

    def __enter__(self) -> Self:
        """Open the bars when a stream was given."""
        if self._file is None:
            return self
        index, count = preset_position()
        file = self._file
        model = self._model

        def bar(
            position: int, total: int, desc: str, layout: str, initial: int = 0
        ) -> "tqdm[Never]":
            return tqdm(
                total=total,
                initial=initial,
                desc=desc,
                bar_format=layout,
                position=position,
                file=file,
                dynamic_ncols=True,
                leave=False,
            )

        self._bars = [
            bar(
                0,
                count,
                f"model    {index}/{count} {model}",
                "{desc} |{bar}| {n_fmt}/{total_fmt} done [{elapsed}]",
                initial=index - 1,
            ),
            bar(
                1,
                self._planned,
                f"runs     {model}",
                "{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                initial=self._finished,
            ),
            bar(2, self._repeats, "repeat  ", "{desc} |{bar}| {n_fmt}/{total_fmt}"),
            bar(3, self._scenarios, "scenario", "{desc} |{bar}| {n_fmt}/{total_fmt} {postfix}"),
        ]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the bars, innermost first."""
        for bar in reversed(self._bars):
            bar.close()
        self._bars = []

    def repeat(self, repeat: int) -> None:
        """A repeat (1-based) starts."""
        if self._bars:
            _set(self._bars[2], repeat - 1)
            self._bars[2].set_description_str(f"repeat   {repeat}/{self._repeats}")

    def scenario(self, index: int, scenario_id: str) -> None:
        """The `index`-th (1-based) scenario of the repeat starts."""
        if self._bars:
            _set(self._bars[3], index - 1)
            self._bars[3].set_description_str(f"scenario {scenario_id}")

    def run(self, level: str, position: int, size: int) -> None:
        """A run of `level` starts at `position` of its block's `size` planned levels."""
        if self._bars:
            self._bars[3].set_postfix_str(f"{level} ({position}/{size})")

    def run_done(self) -> None:
        """The current run finished."""
        if self._bars:
            self._bars[1].update(1)
            self._bars[1].refresh()  # runs are slow; never skip a redraw


def _set(bar: "tqdm[Never]", value: int) -> None:
    bar.n = value
    bar.refresh()
