from pathlib import Path
from typing import Any

import httpx
import pytest

from tool_eval.config import Settings
from tool_eval.exceptions import ExternalServiceError
from tool_eval.services import arxiv, papers, pdf


class _Page:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _Reader:
    def __init__(self, path: Any) -> None:
        self.pages = [_Page("Fraktur \ud835 letter"), _Page("second page")]


def test_lone_surrogates_are_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pdf, "PdfReader", _Reader)
    (tmp_path / "papers").mkdir()
    (tmp_path / "papers" / "2401.00001.pdf").write_bytes(b"%PDF-1.4")
    settings = Settings(_env_file=None, workspace_dir=tmp_path)

    ref = papers.extract_paper_text("papers/2401.00001.pdf", settings=settings)

    text = (tmp_path / ref).read_text(encoding="utf-8")
    assert "Fraktur ? letter" in text
    assert "second page" in text


# --- arXiv cache and retry ------------------------------------------------------------------

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Cached Paper</title>
    <summary>Abstract.</summary>
    <published>2024-01-02T00:00:00Z</published>
    <author><name>Jane Doe</name></author>
  </entry>
</feed>"""


class _Server:
    """Scripted arXiv: returns the given responses in order and counts requests."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return self.responses.pop(0)


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(arxiv, "_MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(arxiv, "_sleep", recorded.append)
    return recorded


def _serve(monkeypatch: pytest.MonkeyPatch, server: _Server) -> None:
    monkeypatch.setattr(
        arxiv,
        "build_client",
        lambda *, timeout: httpx.Client(transport=httpx.MockTransport(server)),
    )


def _fetch(cache_dir: Path | None) -> list[arxiv.ArxivPaper]:
    return arxiv.fetch_arxiv_papers(["2401.00001"], timeout=5, cache_dir=cache_dir)


def test_second_identical_request_is_served_from_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    server = _Server(httpx.Response(200, text=FEED))
    _serve(monkeypatch, server)
    assert _fetch(tmp_path)[0].title == "Cached Paper"
    assert _fetch(tmp_path)[0].title == "Cached Paper"
    assert server.calls == 1


def test_cache_is_read_by_a_fresh_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    _serve(monkeypatch, _Server(httpx.Response(200, text=FEED)))
    _fetch(tmp_path)
    # A new process has no in-memory state; only the files on disk remain.
    offline = _Server()
    _serve(monkeypatch, offline)
    assert _fetch(tmp_path)[0].arxiv_id == "2401.00001"
    assert offline.calls == 0


def test_failed_requests_are_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    _serve(monkeypatch, _Server(*[httpx.Response(503)] * 5))
    with pytest.raises(ExternalServiceError):
        _fetch(tmp_path)
    assert not list(tmp_path.iterdir())


def test_rate_limit_is_retried_honouring_retry_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    server = _Server(
        httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, text=FEED)
    )
    _serve(monkeypatch, server)
    assert _fetch(None)[0].title == "Cached Paper"
    assert server.calls == 2
    assert waits and waits[0] >= 7


def test_persistent_rate_limit_fails_after_five_attempts(
    monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    server = _Server(*[httpx.Response(429)] * 5)
    _serve(monkeypatch, server)
    with pytest.raises(ExternalServiceError, match="arXiv"):
        _fetch(None)
    assert server.calls == 5


def test_client_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch, waits: list[float]) -> None:
    server = _Server(httpx.Response(400))
    _serve(monkeypatch, server)
    with pytest.raises(ExternalServiceError):
        _fetch(None)
    assert server.calls == 1
    assert waits == []


def test_timeouts_are_retried(monkeypatch: pytest.MonkeyPatch, waits: list[float]) -> None:
    calls = 0

    def flaky(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, text=FEED)

    monkeypatch.setattr(
        arxiv, "build_client", lambda *, timeout: httpx.Client(transport=httpx.MockTransport(flaky))
    )
    assert _fetch(None)[0].arxiv_id == "2401.00001"
    assert calls == 2
