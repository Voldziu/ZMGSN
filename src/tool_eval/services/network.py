"""Run-scoped count of arXiv requests and PDF downloads that missed the caches.

`runner.run_single` opens a meter per run; outside a meter, counting does nothing.
"""

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

NetworkKind = Literal["arxiv_requests", "pdf_downloads"]

_meter: ContextVar[Counter[str] | None] = ContextVar("network_meter", default=None)


@contextmanager
def network_meter() -> Iterator[Counter[str]]:
    """Count network requests inside the `with` block (including tool threads).

    Yields:
        Counter with `arxiv_requests` and `pdf_downloads`, both starting at 0.
    """
    counts: Counter[str] = Counter(arxiv_requests=0, pdf_downloads=0)
    token = _meter.set(counts)
    try:
        yield counts
    finally:
        _meter.reset(token)


def count_network_request(kind: NetworkKind) -> None:
    """Count one request of `kind` that is about to reach the network."""
    counts = _meter.get()
    if counts is not None:
        counts[kind] += 1
