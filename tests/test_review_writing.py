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
    assert result.complete is False
    assert "Only 3 of 5 papers" in result.message
    assert "incomplete" in result.message


def test_level3_states_that_the_task_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _level3(tmp_path, monkeypatch, found=10, unreadable=set())
    assert result.complete is True
    assert result.message == (
        "Review with 5 papers read in full saved to reviews/review.md. The task is complete."
    )


def test_a_repeated_level3_call_starts_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = tmp_path / "texts" / "1999.99999.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("from an earlier call", encoding="utf-8")
    _level3(tmp_path, monkeypatch, found=10, unreadable=set())
    assert not stale.exists()


# --- level 1 writing and tool descriptions ---------------------------------------------------


def test_level1_writes_by_replacing_and_keeps_its_distractors(tmp_path: Path) -> None:
    from tool_eval.config import use_settings
    from tool_eval.tools.level1.files import write_file

    names = {t.name for t in get_tools("level1")}
    assert "write_file" in names
    assert not {n for n in names if "append" in n}
    assert {"search_wikipedia", "extract_keywords", "translate_text"} <= names
    assert "create_markdown_table" in names
    with use_settings(Settings(_env_file=None, workspace_dir=tmp_path)):
        write_file.invoke({"path": "reviews/review.md", "content": "first draft"})
        write_file.invoke({"path": "reviews/review.md", "content": "corrected"})
    assert (tmp_path / "reviews" / "review.md").read_text(encoding="utf-8") == "corrected"


def test_no_tool_description_shows_a_real_looking_arxiv_id() -> None:
    import re

    for level in ("level1", "level2", "level3"):
        for tool in get_tools(level):
            schema = str(tool.args)
            assert not re.search(r"\d{4}\.\d{4,5}", tool.description + schema), tool.name


def test_task_prompt_names_no_tool_step_but_requires_full_text() -> None:
    from tool_eval.prompts import load_prompt

    text = load_prompt("evaluation", "v3").render(
        topic="t", n_papers="5", date_from="2024-01-01", date_to="2024-06-30"
    )
    for word in ("download", "extract", "split", "summarize"):
        assert word not in text.lower()
    assert "full text" in text


def test_level2_and_level3_summaries_are_synthesized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_core.language_models import FakeListChatModel

    from tool_eval.services import papers

    text = " ".join(f"w{i:05d}" for i in range(3000))
    monkeypatch.setattr(papers, "download_paper_pdf", lambda arxiv_id, **_: "papers/x.pdf")
    monkeypatch.setattr(
        papers,
        "extract_paper_text",
        lambda ref, **_: _write(tmp_path, "texts/2401.00001.txt", text),
    )
    llm = FakeListChatModel(responses=[f"part {i}" for i in range(6)] + ["one short synthesis"])
    result = papers.summarize_paper(
        "2401.00001",
        settings=Settings(_env_file=None, workspace_dir=tmp_path, prompt_version="v3"),
        llm=llm,
    )
    assert result.summary == "one short synthesis"
    assert result.chunks_summarized == 6
    assert (tmp_path / result.summary_path).read_text(encoding="utf-8") == "one short synthesis"


def _write(root: Path, ref: str, text: str) -> str:
    path = root / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return ref
