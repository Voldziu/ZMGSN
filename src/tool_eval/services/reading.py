"""Run-scoped record of how much of each paper's extracted text was read.

Text counts as read when a tool hands it to the agent (`read_file`) or passes it to a model
(chunk summaries, keywords, translation, the level-2/3 pipeline). Each reading path records the
span `[start, end)` of `texts/<stem>.txt` it covered; overlapping spans are counted once.

`runner.run_single` opens a meter per run with `reading_meter()`; outside a meter, recording does
nothing, so tools work unchanged in interactive use.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from tool_eval.utils.paths import to_ref

CHUNK_INDEX = "index.json"


class ReadMeter:
    """Spans read per paper during one run."""

    def __init__(self) -> None:
        """Start with nothing read."""
        self._spans: dict[str, list[tuple[int, int]]] = {}

    def add(self, stem: str, start: int, end: int) -> None:
        """Record that characters `[start, end)` of paper `stem` were read."""
        if end > start:
            self._spans.setdefault(stem, []).append((start, end))

    def chars_read(self) -> dict[str, int]:
        """Distinct characters read per paper stem."""
        return {stem: _union_length(spans) for stem, spans in sorted(self._spans.items())}


_meter: ContextVar[ReadMeter | None] = ContextVar("read_meter", default=None)


@contextmanager
def reading_meter() -> Iterator[ReadMeter]:
    """Record reads into a fresh meter inside the `with` block (including tool threads).

    Yields:
        The meter.
    """
    meter = ReadMeter()
    token = _meter.set(meter)
    try:
        yield meter
    finally:
        _meter.reset(token)


def record_text_read(path: Path, *, workspace: Path, kept: int) -> None:
    """Record that the first `kept` characters of the file at `path` were read.

    Paper texts (`texts/<stem>.txt`) and chunk files (`chunks/<stem>/NNNN.txt`, mapped back to
    their span in the text through the chunk index) count; any other file is ignored.

    Args:
        path: File that was read.
        workspace: Run workspace.
        kept: Characters of the file that were passed on (after any cut).
    """
    meter = _meter.get()
    if meter is None:
        return
    parts = Path(to_ref(path, workspace=workspace)).parts
    if len(parts) == 2 and parts[0] == "texts" and parts[1].endswith(".txt"):
        meter.add(parts[1].removesuffix(".txt"), 0, kept)
    elif len(parts) == 3 and parts[0] == "chunks":
        span = _chunk_span(path.parent / CHUNK_INDEX, parts[2])
        if span is not None:
            meter.add(parts[1], span[0], min(span[1], span[0] + kept))


def write_chunk_index(directory: Path, spans: dict[str, tuple[int, int]]) -> None:
    """Store the span of every chunk file of a paper next to the chunks.

    Args:
        directory: `chunks/<stem>/` directory.
        spans: Chunk file name to its `[start, end)` span in the paper text.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CHUNK_INDEX).write_text(json.dumps(spans), encoding="utf-8")


def _chunk_span(index: Path, name: str) -> tuple[int, int] | None:
    try:
        spans = json.loads(index.read_text(encoding="utf-8"))
        start, end = spans[name]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return int(start), int(end)


def _union_length(spans: list[tuple[int, int]]) -> int:
    total, reach = 0, -1
    for start, end in sorted(spans):
        if end <= reach:
            continue
        total += end - max(start, reach)
        reach = end
    return total
