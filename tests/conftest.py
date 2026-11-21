import pytest

from tool_eval.evaluation import runner


@pytest.fixture(autouse=True)
def _no_scoring_waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Scoring retries never sleep in tests; the waits are recorded instead."""
    waits: list[float] = []
    monkeypatch.setattr(runner, "_sleep", waits.append)
    return waits
