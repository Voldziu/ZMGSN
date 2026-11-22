from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from test_evaluation import SCENARIO, _lookup, _mark_read, _papers, _record, _review_text
from tool_eval.evaluation.judge import (
    ReviewRating,
    SummaryRating,
    SurveyDecision,
    judge_results,
    judge_run,
    judgment_file,
    load_judgments,
)
from tool_eval.evaluation.models import CriterionResult, RunRecord, Scenario
from tool_eval.services.arxiv import ArxivPaper


class _FakeJudge:
    """Answers every schema with fixed values and records every prompt."""

    def __init__(self, *, rating: int = 5, survey: bool = False, broken: int = 0) -> None:
        self.prompts: list[str] = []
        self.rating = rating
        self.survey = survey
        self.broken = broken  # number of calls that return output not fitting the schema

    def __call__(self, schema: type[BaseModel], system: str, user: str) -> BaseModel:
        self.prompts.append(system + "\n" + user)
        if self.broken:
            self.broken -= 1
            raise ValueError("not JSON")
        if schema is ReviewRating:
            return ReviewRating(valid_review=self.rating, relevance=self.rating)
        if schema is SummaryRating:
            return SummaryRating(faithfulness=self.rating)
        return SurveyDecision(is_survey=self.survey)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "runs" / "level3-demo-01"
    review = workspace / "reviews" / "review.md"
    review.parent.mkdir(parents=True)
    review.write_text(_review_text(), encoding="utf-8")
    _mark_read(workspace, [f"2401.0000{i}" for i in range(5)])
    return workspace


def _run_record(run_id: str = "level3-demo-01", **update: Any) -> RunRecord:
    record = _record("level3", "demo", 1, score=1.0, tokens=1)
    assert record.outcome is not None
    outcome = record.outcome.model_copy(
        update={"facts": [CriterionResult(name="papers_cited", passed=True)]}
    )
    return record.model_copy(
        update={"run_id": run_id, "outcome": outcome, "config_sha256": "cfg", **update}
    )


def _judge(tmp_path: Path, fake: _FakeJudge, scenario: Scenario = SCENARIO) -> Any:
    return judge_run(
        _run_record(),
        workspace=_workspace(tmp_path),
        scenario=scenario,
        requirement_statement=None,
        ask=fake,
        lookup=_lookup(_papers()),
        judge_sha256="j",
    )


def test_every_rating_is_a_number(tmp_path: Path) -> None:
    judgment = _judge(tmp_path, _FakeJudge(rating=4))
    assert (judgment.valid_review, judgment.relevance) == (4, 4)
    assert judgment.faithfulness == [4, 4]
    assert judgment.surveys is None
    assert judgment.passes is True


def test_the_judge_is_blind_to_model_level_and_run(tmp_path: Path) -> None:
    fake = _FakeJudge()
    _judge(tmp_path, fake)
    text = "\n".join(fake.prompts)
    assert "level3" not in text
    assert "level3-demo-01" not in text
    assert "qwen" not in text.lower()
    assert SCENARIO.topic in fake.prompts[0]


def test_the_same_run_always_gets_the_same_sections(tmp_path: Path) -> None:
    first, second = _FakeJudge(), _FakeJudge()
    _judge(tmp_path / "a", first)
    _judge(tmp_path / "b", second)
    assert first.prompts == second.prompts


def test_an_answer_that_never_fits_is_missing(tmp_path: Path) -> None:
    judgment = _judge(tmp_path, _FakeJudge(broken=2))
    assert judgment.valid_review is None
    assert judgment.passes is None


def test_one_bad_answer_is_retried(tmp_path: Path) -> None:
    assert _judge(tmp_path, _FakeJudge(broken=1)).valid_review == 5


def test_surveys_are_decided_for_no_surveys_variants(tmp_path: Path) -> None:
    variant = Scenario(
        id="demo-no_surveys",
        topic=SCENARIO.topic,
        date_from=SCENARIO.date_from,
        date_to=SCENARIO.date_to,
        requirement="no_surveys",
        base="demo",
    )
    judgment = _judge(tmp_path, _FakeJudge(survey=True), variant)
    assert judgment.surveys == {f"2401.0000{i}": True for i in range(5)}


def test_missing_review_gets_no_rating(tmp_path: Path) -> None:
    judgment = judge_run(
        _run_record(),
        workspace=tmp_path,
        scenario=SCENARIO,
        requirement_statement=None,
        ask=_FakeJudge(),
        lookup=_lookup(_papers()),
        judge_sha256="j",
    )
    assert judgment.valid_review is None
    assert judgment.faithfulness == []


def test_results_are_judged_once_and_resumed(tmp_path: Path) -> None:
    _workspace(tmp_path)
    output = tmp_path / "presets-m.jsonl"
    errored = _run_record("level3-other-01", status="error", outcome=None)
    output.write_text(
        _run_record().model_dump_json() + "\n" + errored.model_dump_json() + "\n",
        encoding="utf-8",
    )

    def judge(fake: _FakeJudge, sha: str) -> tuple[int, int]:
        return judge_results(
            output,
            runs_dir=tmp_path / "runs",
            scenarios={"demo": SCENARIO},
            requirement_statement=lambda s: None,
            ask=fake,
            lookup=_lookup(_papers()),
            judge_sha256=sha,
        )

    assert judge(_FakeJudge(), "j1") == (1, 0)
    assert judge(_FakeJudge(), "j1") == (0, 1)
    assert judge(_FakeJudge(rating=2), "j2") == (1, 0)  # a new judge configuration rejudges
    stored = load_judgments(judgment_file(output))
    assert judgment_file(output).name == "presets-m.judge.jsonl"
    assert [j.judge_sha256 for j in stored] == ["j1", "j2"]


def test_lookup_without_the_paper_leaves_the_survey_decision_open(tmp_path: Path) -> None:
    variant = Scenario(
        id="demo-no_surveys",
        topic=SCENARIO.topic,
        date_from=date(2024, 1, 1),
        date_to=date(2024, 12, 31),
        requirement="no_surveys",
        base="demo",
    )
    known: dict[str, ArxivPaper] = {}
    judgment = judge_run(
        _run_record(),
        workspace=_workspace(tmp_path),
        scenario=variant,
        requirement_statement="Additional requirement: no surveys.",
        ask=_FakeJudge(),
        lookup=_lookup(known),
        judge_sha256="j",
    )
    assert judgment.surveys is not None
    assert set(judgment.surveys.values()) == {None}
