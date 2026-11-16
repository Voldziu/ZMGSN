from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import GenericFakeChatModel

from tool_eval.config import Settings
from tool_eval.exceptions import ExternalServiceError
from tool_eval.services import reports
from tool_eval.services.arxiv import ArxivPaper
from tool_eval.services.papers import PaperSummary
from tool_eval.tools.registry import get_tools

PAPERS = {
    f"2401.0000{i}": ArxivPaper(
        arxiv_id=f"2401.0000{i}",
        title=f"Paper {i}",
        authors=["Jane Doe"],
        published=date(2024, 3, 1),
        abstract="",
    )
    for i in range(3)
}


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    def fetch(ids: list[str], **kwargs: Any) -> list[ArxivPaper]:
        return [PAPERS[i] for i in ids]

    monkeypatch.setattr(reports, "fetch_arxiv_papers", fetch)
    return Settings(_env_file=None, workspace_dir=tmp_path)


def _review(settings: Settings) -> str:
    return (settings.workspace_dir / reports.REVIEW_PATH).read_text(encoding="utf-8")


def test_level2_offers_write_review_and_no_append() -> None:
    names = {t.name for t in get_tools("level2")}
    assert "write_review" in names
    assert "append_to_review" not in names


def test_write_review_replaces_the_file(settings: Settings) -> None:
    reports.write_review("# First\n\n## 1. Old", settings=settings)
    reports.write_review("# Second\n\n## 1. New", settings=settings)
    assert _review(settings) == "# Second\n\n## 1. New\n"


def test_bibliography_written_twice_keeps_one_section(settings: Settings) -> None:
    reports.write_review("# Review\n\n## 1. Paper 0", settings=settings)
    reports.append_bibliography(["2401.00000", "2401.00001"], settings=settings)
    reports.append_bibliography(["2401.00002"], settings=settings)

    text = _review(settings)
    assert text.count("## Bibliography") == 1
    assert "Paper 2" in text
    assert "Paper 1" not in text
    assert text.startswith("# Review\n\n## 1. Paper 0\n\n## Bibliography")


def test_bibliography_without_review_creates_the_file(settings: Settings) -> None:
    reports.append_bibliography(["2401.00000"], settings=settings)
    assert _review(settings).startswith("## Bibliography")


def test_rewriting_the_review_drops_the_bibliography(settings: Settings) -> None:
    reports.write_review("# Review", settings=settings)
    reports.append_bibliography(["2401.00000"], settings=settings)
    reports.write_review("# Review v2", settings=settings)
    assert _review(settings) == "# Review v2\n"


# --- level 3: spare candidates ----------------------------------------------------------------


def _candidates(count: int) -> list[ArxivPaper]:
    return [
        ArxivPaper(
            arxiv_id=f"2401.000{i:02d}",
            title=f"Candidate {i}",
            authors=["Jane Doe"],
            published=date(2024, 3, 1),
            abstract="",
        )
        for i in range(count)
    ]


def _level3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, found: int, unreadable: set[int]
) -> tuple[reports.ReviewResult, list[int]]:
    asked: list[int] = []

    def search(query: str, *, max_results: int, **kwargs: Any) -> list[ArxivPaper]:
        asked.append(max_results)
        return _candidates(found)[:max_results]

    def summarize(arxiv_id: str, **kwargs: Any) -> PaperSummary:
        if int(arxiv_id[-2:]) in unreadable:
            raise ExternalServiceError(f"The PDF of {arxiv_id} exceeds the 100 MB size limit.")
        return PaperSummary(
            arxiv_id=arxiv_id,
            summary_path=f"summaries/{arxiv_id}.md",
            summary="A concise summary that is clearly long enough to count as one.",
            chunks_summarized=1,
            chunks_total=1,
        )

    monkeypatch.setattr(reports, "search_arxiv", search)
    monkeypatch.setattr(reports, "summarize_paper", summarize)
    settings = Settings(_env_file=None, workspace_dir=tmp_path)
    result = reports.conduct_review(
        "demo", settings=settings, llm=GenericFakeChatModel(messages=iter([]))
    )
    return result, asked


def test_level3_replaces_an_unreadable_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, asked = _level3(tmp_path, monkeypatch, found=10, unreadable={2})
    assert asked == [10]
    assert result.papers_included == 5
    assert [s.arxiv_id for s in result.papers_skipped] == ["2401.00002"]
    assert "100 MB" in result.papers_skipped[0].reason
    text = (tmp_path / reports.REVIEW_PATH).read_text(encoding="utf-8")
    assert "Candidate 5" in text
    assert "Candidate 2" not in text


def test_level3_reports_a_shortfall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = _level3(tmp_path, monkeypatch, found=6, unreadable={0, 1, 4})
    assert result.papers_included == 3
    assert result.papers_requested == 5
    assert len(result.papers_skipped) == 3
