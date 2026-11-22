"""Fact checks, requirement checks and the three outcomes (artifact valid, reliable, success)."""

from pathlib import Path

import pytest

from test_evaluation import SCENARIO, _lookup, _mark_read, _papers, _record, _review_text
from tool_eval.evaluation.checks import cited_ids, evaluate_review
from tool_eval.evaluation.models import (
    CriterionResult,
    Judgment,
    OutcomeScore,
    Requirement,
    RunRecord,
    Scenario,
)

IDS = [f"2401.0000{i}" for i in range(5)]


def _score(
    tmp_path: Path, text: str, *, read: list[str] | None = None, scenario: Scenario = SCENARIO
) -> OutcomeScore:
    review = tmp_path / "reviews" / "review.md"
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text(text, encoding="utf-8")
    _mark_read(tmp_path, IDS if read is None else read)
    return evaluate_review(
        review,
        workspace=tmp_path,
        scenario=scenario,
        expected_papers=5,
        lookup=_lookup(_papers()),
    )


def _facts(score: OutcomeScore) -> dict[str, bool]:
    return {f.name: f.passed for f in score.facts}


def _variant(requirement: Requirement) -> Scenario:
    return Scenario(
        id=f"demo-{requirement}",
        topic=SCENARIO.topic,
        date_from=SCENARIO.date_from,
        date_to=SCENARIO.date_to,
        requirement=requirement,
        base="demo",
    )


# --- fact checks ------------------------------------------------------------------------------


def test_ids_are_found_anywhere_whatever_the_format() -> None:
    text = (
        "see https://arxiv.org/abs/2401.00001v2 and arXiv:2401.00002, "
        "https://arxiv.org/pdf/2401.00003 and again https://arxiv.org/abs/2401.00001"
    )
    assert cited_ids(text) == ["2401.00001", "2401.00002", "2401.00003"]


def test_wrong_heading_level_fails_format_but_not_facts(tmp_path: Path) -> None:
    score = _score(tmp_path, _review_text().replace("## ", "### "))
    assert all(_facts(score).values())
    assert not next(c for c in score.criteria if c.name == "section_count").passed


def test_one_line_file_with_literal_newlines_still_has_its_facts(tmp_path: Path) -> None:
    score = _score(tmp_path, _review_text().replace("\n", "\\n"))
    assert all(_facts(score).values())
    assert score.score < 0.5


def test_ids_only_in_the_bibliography_count(tmp_path: Path) -> None:
    text = "# Review\n\nSummaries without links.\n\n## Bibliography\n" + "\n".join(
        f"- https://arxiv.org/abs/{i}" for i in IDS
    )
    assert all(_facts(_score(tmp_path, text)).values())


def test_invented_id_fails_existence_and_extraction(tmp_path: Path) -> None:
    text = _review_text().replace("2401.00004", "2401.99999")
    facts = _facts(_score(tmp_path, text, read=IDS[:4]))
    assert facts["papers_cited"]
    assert not facts["cited_ids_exist"]
    assert not facts["text_extracted"]


def test_too_few_papers(tmp_path: Path) -> None:
    assert not _facts(_score(tmp_path, _review_text(3)))["papers_cited"]


def test_missing_file_fails_every_fact(tmp_path: Path) -> None:
    score = evaluate_review(
        tmp_path / "reviews" / "review.md",
        workspace=tmp_path,
        scenario=_variant("table"),
        expected_papers=5,
        lookup=_lookup(_papers()),
    )
    assert not any(_facts(score).values())
    assert score.requirement is not None
    assert not score.requirement.passed


# --- requirement checks -----------------------------------------------------------------------


def _with_table(rows: list[str], *, after_first_section: bool = False) -> str:
    table = ["| title | year | contribution |", "|---|---|---|", *rows]
    text = _review_text()
    if after_first_section:
        head, rest = text.split("## 2.", 1)
        return head + "\n".join(table) + "\n\n## 2." + rest
    title, rest = text.split("\n", 1)
    return f"{title}\n\n" + "\n".join(table) + "\n" + rest


def test_table_in_place_passes(tmp_path: Path) -> None:
    rows = [f"| Paper Title {i} | 2024 | x |" for i in range(5)]
    score = _score(tmp_path, _with_table(rows), scenario=_variant("table"))
    assert score.requirement is not None
    assert score.requirement.passed


def test_table_rows_may_name_the_id(tmp_path: Path) -> None:
    rows = [f"| {i} | 2024 | x |" for i in IDS]
    score = _score(tmp_path, _with_table(rows), scenario=_variant("table"))
    assert score.requirement is not None
    assert score.requirement.passed


@pytest.mark.parametrize(
    "text",
    [
        _review_text(),  # no table
        _with_table([f"| Paper Title {i} | 2024 | x |" for i in range(4)]),  # one paper missing
        _with_table(
            [f"| Paper Title {i} | 2024 | x |" for i in range(5)], after_first_section=True
        ),
    ],
)
def test_table_requirement_failures(tmp_path: Path, text: str) -> None:
    score = _score(tmp_path, text, scenario=_variant("table"))
    assert score.requirement is not None
    assert not score.requirement.passed


POLISH = "Autorzy proponują nową metodę wyszukiwania dokumentów, która poprawia jakość odpowiedzi."


def test_polish_summaries_pass_with_english_titles_and_references(tmp_path: Path) -> None:
    text = _review_text().replace(
        "This paper studies the topic and reports clear, reproducible results.", POLISH
    )
    score = _score(tmp_path, text, scenario=_variant("polish"))
    assert score.requirement is not None
    assert score.requirement.passed, score.requirement.detail


def test_english_summaries_fail_the_polish_requirement(tmp_path: Path) -> None:
    score = _score(tmp_path, _review_text(), scenario=_variant("polish"))
    assert score.requirement is not None
    assert not score.requirement.passed
    assert "en" in score.requirement.detail


def test_base_and_no_surveys_tasks_have_no_deterministic_requirement(tmp_path: Path) -> None:
    assert _score(tmp_path, _review_text()).requirement is None
    assert _score(tmp_path, _review_text(), scenario=_variant("no_surveys")).requirement is None


# --- outcomes ---------------------------------------------------------------------------------


def _with_outcome(record: RunRecord, *, facts: bool = True, **update: object) -> RunRecord:
    assert record.outcome is not None
    outcome = record.outcome.model_copy(
        update={"facts": [CriterionResult(name="papers_cited", passed=facts)]}
    )
    return record.model_copy(update={"outcome": outcome, **update})


def _judgment(rating: int = 5, **update: object) -> Judgment:
    data: dict[str, object] = {
        "run_id": "r",
        "config_sha256": "c",
        "judge_sha256": "j",
        "valid_review": rating,
        "relevance": rating,
        "faithfulness": [rating, rating],
    }
    return Judgment.model_validate(data | update)


def test_loop_after_a_valid_review_is_valid_but_not_reliable() -> None:
    record = _with_outcome(_record("level3", "a", 1, score=1.0, tokens=1, status="loop"))
    assert record.artifact_valid(_judgment()) is True
    assert not record.reliable
    assert record.success_with(_judgment()) is False


def test_valid_content_with_wrong_format_is_a_success_but_not_a_format_success() -> None:
    record = _with_outcome(_record("level1", "a", 1, score=0.1, tokens=1))
    assert not record.format_success
    assert record.success_with(_judgment()) is True


def test_low_rating_or_failed_fact_is_not_valid() -> None:
    record = _with_outcome(_record("level1", "a", 1, score=1.0, tokens=1))
    assert record.artifact_valid(_judgment(3)) is False
    assert _with_outcome(record, facts=False).artifact_valid(_judgment()) is False


def test_missing_judgment_or_rating_leaves_validity_undecided() -> None:
    record = _with_outcome(_record("level1", "a", 1, score=1.0, tokens=1))
    assert record.artifact_valid(None) is None
    assert record.artifact_valid(_judgment(relevance=None)) is None
    assert record.success_with(None) is None


def test_failed_facts_decide_without_a_judgment() -> None:
    record = _with_outcome(_record("level1", "a", 1, score=1.0, tokens=1), facts=False)
    assert record.artifact_valid(None) is False


def test_no_surveys_variant_is_decided_by_the_judge() -> None:
    record = _with_outcome(
        _record("level2", "a-no_surveys", 1, score=1.0, tokens=1),
        task_type="variant",
        requirement="no_surveys",
    )
    assert record.artifact_valid(_judgment()) is None  # surveys not judged yet
    assert record.artifact_valid(_judgment(surveys={"x": False, "y": False})) is True
    assert record.artifact_valid(_judgment(surveys={"x": True, "y": False})) is False


def test_old_records_have_undecided_validity() -> None:
    record = _record("level1", "a", 1, score=1.0, tokens=1)
    assert record.artifact_valid(_judgment()) is None
    assert record.format_success


def test_review_without_rateable_sections_is_not_valid() -> None:
    record = _with_outcome(_record("level2", "a", 1, score=0.1, tokens=1))
    assert record.artifact_valid(_judgment(faithfulness=[])) is False
