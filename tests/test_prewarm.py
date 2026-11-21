from datetime import date
from pathlib import Path
from typing import Any

import pytest

from tool_eval.config import Settings
from tool_eval.evaluation.models import Scenario
from tool_eval.exceptions import PaperNotFoundError
from tool_eval.services import papers
from tool_eval.services import prewarm as module
from tool_eval.services.arxiv import ArxivPaper


def _paper(i: int) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=f"2401.0000{i}",
        title=f"P{i}",
        authors=["A B"],
        published=date(2024, 2, 1),
        abstract="",
    )


def test_prewarm_searches_each_topic_once_and_downloads_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    searches: list[tuple[str, int]] = []
    downloads: list[str] = []

    def search(query: str, *, max_results: int, **_: Any) -> list[ArxivPaper]:
        searches.append((query, max_results))
        return [_paper(i) for i in range(3)]

    def download(arxiv_id: str, **_: Any) -> str:
        downloads.append(arxiv_id)
        if arxiv_id.endswith("2"):
            raise PaperNotFoundError("arXiv returned HTTP 404")
        return f"papers/{arxiv_id}.pdf"

    monkeypatch.setattr(module, "search_arxiv", search)
    monkeypatch.setattr(module, "fetch_arxiv_papers", lambda ids, **_: [])
    monkeypatch.setattr(papers, "download_paper_pdf", download)
    base = Scenario(id="a", topic="rag", date_from=date(2024, 1, 1), date_to=date(2024, 6, 30))
    twin = Scenario(id="b", topic="rag", date_from=date(2024, 1, 1), date_to=date(2024, 6, 30))
    report = module.prewarm(
        [base, twin], settings=Settings(_env_file=None, pdf_cache_dir=tmp_path / "c")
    )
    assert searches == [("rag", 10)]
    assert downloads == ["2401.00000", "2401.00001", "2401.00002"]
    assert report.candidates == 3
    assert report.missing_pdfs == ["2401.00002"]
    assert report.failed == []
