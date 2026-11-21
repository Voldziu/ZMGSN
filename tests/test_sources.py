from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from tool_eval.config import Settings
from tool_eval.exceptions import ExternalServiceError, PaperNotFoundError
from tool_eval.services import arxiv, papers, pdf
from tool_eval.services.network import network_meter


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


# --- PDF downloads --------------------------------------------------------------------------


def _serve_pdf(monkeypatch: pytest.MonkeyPatch, response: httpx.Response) -> None:
    monkeypatch.setattr(
        pdf,
        "build_client",
        lambda *, timeout: httpx.Client(transport=httpx.MockTransport(lambda request: response)),
    )


def test_pdf_limit_defaults_to_100_mb_and_text_limit_stays() -> None:
    settings = Settings(_env_file=None)
    assert settings.max_pdf_bytes == 100_000_000
    assert settings.max_file_bytes == 20_000_000


def test_large_pdf_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"%PDF-1.4" + b"0" * 30_000_000
    _serve_pdf(monkeypatch, httpx.Response(200, content=body))
    settings = Settings(_env_file=None, workspace_dir=tmp_path)
    ref = papers.download_paper_pdf("2312.09754", settings=settings)
    assert (tmp_path / ref).stat().st_size == len(body)


def test_oversized_pdf_names_paper_and_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_pdf(monkeypatch, httpx.Response(200, content=b"%PDF-1.4" + b"0" * 2_000))
    settings = Settings(_env_file=None, workspace_dir=tmp_path, max_pdf_bytes=1_000)
    with pytest.raises(ExternalServiceError, match=r"2312\.09754.*0\.001 MB"):
        papers.download_paper_pdf("2312.09754", settings=settings)


def test_declared_size_is_checked_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = httpx.Response(200, content=b"%PDF-1.4", headers={"Content-Length": "200000000"})
    _serve_pdf(monkeypatch, response)
    with pytest.raises(ExternalServiceError, match="2312.09754"):
        papers.download_paper_pdf(
            "2312.09754", settings=Settings(_env_file=None, workspace_dir=tmp_path)
        )


def test_missing_pdf_names_paper_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_pdf(monkeypatch, httpx.Response(404, text="not found"))
    monkeypatch.setattr(papers, "fetch_arxiv_papers", lambda ids, **_: [])
    with pytest.raises(ExternalServiceError) as caught:
        papers.download_paper_pdf(
            "2301.00016", settings=Settings(_env_file=None, workspace_dir=tmp_path)
        )
    assert "2301.00016" in str(caught.value)
    assert "HTTP 404" in str(caught.value)


def _paper(version: int | None) -> arxiv.ArxivPaper:
    return arxiv.ArxivPaper(
        arxiv_id="2209.14086",
        title="T",
        authors=["A B"],
        published=date(2022, 9, 1),
        abstract="",
        version=version,
    )


def _serve_by_url(monkeypatch: pytest.MonkeyPatch, pages: dict[str, httpx.Response]) -> list[str]:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return pages.get(str(request.url), httpx.Response(404, text="not found"))

    monkeypatch.setattr(
        pdf, "build_client", lambda *, timeout: httpx.Client(transport=httpx.MockTransport(handler))
    )
    return seen


def test_missing_unversioned_pdf_is_tried_at_its_latest_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _serve_by_url(
        monkeypatch,
        {"https://arxiv.org/pdf/2209.14086v2": httpx.Response(200, content=b"%PDF-1.4 x")},
    )
    monkeypatch.setattr(papers, "fetch_arxiv_papers", lambda ids, **_: [_paper(2)])
    settings = Settings(_env_file=None, workspace_dir=tmp_path / "w", pdf_cache_dir=tmp_path / "c")
    with network_meter() as counts:
        ref = papers.download_paper_pdf("2209.14086", settings=settings)
    assert (tmp_path / "w" / ref).read_bytes().startswith(b"%PDF")
    assert seen == ["https://arxiv.org/pdf/2209.14086", "https://arxiv.org/pdf/2209.14086v2"]
    assert counts["pdf_downloads"] == 2


def test_missing_pdf_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _serve_by_url(monkeypatch, {})
    monkeypatch.setattr(papers, "fetch_arxiv_papers", lambda ids, **_: [_paper(1)])
    settings = Settings(_env_file=None, workspace_dir=tmp_path / "w", pdf_cache_dir=tmp_path / "c")
    for _ in range(2):
        with pytest.raises(PaperNotFoundError, match="HTTP 404"):
            papers.download_paper_pdf("2209.14086", settings=settings)
    assert len(seen) == 2  # the second call is answered from the cache
    assert (tmp_path / "c" / "2209.14086.missing").is_file()


def test_other_pdf_failures_are_not_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_pdf(monkeypatch, httpx.Response(503, text="busy"))
    settings = Settings(_env_file=None, workspace_dir=tmp_path / "w", pdf_cache_dir=tmp_path / "c")
    with pytest.raises(ExternalServiceError):
        papers.download_paper_pdf("2209.14086", settings=settings)
    assert not list((tmp_path / "c").glob("*.missing"))


def test_cached_pdf_needs_no_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "c"
    cache.mkdir()
    (cache / "2209.14086.pdf").write_bytes(b"%PDF-1.4 cached")
    settings = Settings(_env_file=None, workspace_dir=tmp_path / "w", pdf_cache_dir=cache)
    with network_meter() as counts:
        papers.download_paper_pdf("2209.14086", settings=settings)
    assert counts["pdf_downloads"] == 0


def test_entry_version_is_parsed() -> None:
    assert arxiv._parse_feed(FEED)[0].version is not None


# --- search order ---------------------------------------------------------------------------


def test_search_is_sorted_by_relevance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, text=FEED)

    monkeypatch.setattr(
        arxiv,
        "build_client",
        lambda *, timeout: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    arxiv.search_arxiv(
        "protein structure prediction",
        max_results=5,
        date_from=None,
        date_to=None,
        timeout=5,
        cache_dir=tmp_path,
    )
    params = seen[0].params
    assert params["sortBy"] == "relevance"
    assert params["sortOrder"] == "descending"
    old = {**dict(params), "sortBy": "submittedDate"}
    assert arxiv._cache_key(dict(params)) != arxiv._cache_key(old)
