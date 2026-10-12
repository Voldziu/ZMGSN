from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest
from langchain.agents import create_agent
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from tool_eval.config import Settings, get_settings, use_settings
from tool_eval.evaluation import runner
from tool_eval.evaluation.checks import evaluate_review
from tool_eval.evaluation.cost import RunMeter
from tool_eval.evaluation.models import (
    CostMetrics,
    CriterionResult,
    OutcomeScore,
    RunRecord,
    RunStatus,
    Scenario,
)
from tool_eval.evaluation.report import load_records, render_markdown, summarize
from tool_eval.evaluation.runner import run_evaluation
from tool_eval.evaluation.scenarios import load_scenarios
from tool_eval.exceptions import ExternalServiceError, InvalidInputError
from tool_eval.main import main
from tool_eval.prompts import load_prompt
from tool_eval.services import reports
from tool_eval.services.arxiv import ArxivPaper
from tool_eval.services.papers import PaperSummary, split_paper_text
from tool_eval.tools.registry import get_tools

SCENARIO = Scenario(
    id="demo",
    topic="demo topic",
    date_from=date(2024, 1, 1),
    date_to=date(2024, 6, 30),
)


class _ToolCallingFake(GenericFakeChatModel):
    """Scripted chat model that accepts `bind_tools` so `create_agent` can use it."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ToolCallingFake":
        return self


def _settings(workspace: Path, **extra: Any) -> Settings:
    return Settings(_env_file=None, workspace_dir=workspace, agent_temperature=0.7, **extra)


def _papers(count: int = 5, published: date = date(2024, 3, 1)) -> dict[str, ArxivPaper]:
    return {
        f"2401.0000{i}": ArxivPaper(
            arxiv_id=f"2401.0000{i}",
            title=f"Paper Title {i}",
            authors=["Jane Doe"],
            published=published,
            abstract="abstract",
        )
        for i in range(count)
    }


def _lookup(papers: Mapping[str, ArxivPaper]) -> Callable[[Sequence[str]], Mapping[str, Any]]:
    return lambda ids: {i: papers.get(i) for i in ids}


def _review_text(count: int = 5) -> str:
    lines = ["# Review: demo topic", ""]
    references = []
    for i in range(count):
        reference = f"Doe, J. (2024). Paper Title {i}. arXiv. https://arxiv.org/abs/2401.0000{i}"
        references.append(f"- {reference}")
        lines += [
            f"## {i + 1}. Paper Title {i}",
            reference,
            "",
            "This paper studies the topic and reports clear, reproducible results.",
            "",
        ]
    return "\n".join([*lines, "## Bibliography", *references]) + "\n"


def _write_review(workspace: Path, text: str) -> None:
    target = workspace / "reviews" / "review.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


# --- scenarios and prompt ---------------------------------------------------


def test_shipped_scenarios_are_valid() -> None:
    scenarios = load_scenarios(Path(__file__).parent.parent / "scenarios" / "scenarios.json")
    assert len(scenarios) == 10
    assert len({s.id for s in scenarios}) == 10


def test_scenarios_reject_duplicate_ids_and_bad_windows(tmp_path: Path) -> None:
    duplicate = tmp_path / "dup.json"
    one = '{"id": "a", "topic": "topic", "date_from": "2024-01-01", "date_to": "2024-02-01"}'
    duplicate.write_text(f"[{one}, {one}]", encoding="utf-8")
    with pytest.raises(InvalidInputError):
        load_scenarios(duplicate)

    reversed_window = tmp_path / "rev.json"
    reversed_window.write_text(
        '[{"id": "a", "topic": "topic", "date_from": "2024-03-01", "date_to": "2024-02-01"}]',
        encoding="utf-8",
    )
    with pytest.raises(InvalidInputError):
        load_scenarios(reversed_window)


def test_evaluation_prompt_renders_all_placeholders() -> None:
    text = load_prompt("evaluation", "v1").render(
        topic="rag",
        n_papers="5",
        date_from="2024-01-01",
        date_to="2024-06-30",
    )
    assert "$" not in text
    assert "reviews/review.md" in text


# --- outcome checks -----------------------------------------------------------


def test_valid_review_passes_all_checks(tmp_path: Path) -> None:
    _write_review(tmp_path, _review_text())
    score = evaluate_review(
        tmp_path / "reviews/review.md",
        scenario=SCENARIO,
        expected_papers=5,
        lookup=_lookup(_papers()),
    )
    assert score.success, [c for c in score.criteria if not c.passed]
    assert score.score == 1.0


def test_missing_file_fails_every_criterion(tmp_path: Path) -> None:
    score = evaluate_review(
        tmp_path / "reviews/review.md",
        scenario=SCENARIO,
        expected_papers=5,
        lookup=_lookup({}),
    )
    assert not score.success
    assert score.score == 0.0


def test_out_of_window_and_wrong_count_are_detected(tmp_path: Path) -> None:
    _write_review(tmp_path, _review_text(count=4))
    score = evaluate_review(
        tmp_path / "reviews/review.md",
        scenario=SCENARIO,
        expected_papers=5,
        lookup=_lookup(_papers(published=date(2020, 1, 1))),
    )
    failed = {c.name for c in score.criteria if not c.passed}
    assert {"section_count", "dates_in_window"} <= failed
    assert "file_exists" not in failed


def test_unknown_arxiv_id_is_detected(tmp_path: Path) -> None:
    _write_review(tmp_path, _review_text())
    score = evaluate_review(
        tmp_path / "reviews/review.md",
        scenario=SCENARIO,
        expected_papers=5,
        lookup=_lookup({}),
    )
    assert "ids_exist" in {c.name for c in score.criteria if not c.passed}


# --- settings override and cost metering inside a real agent loop -----------


def test_override_and_nested_llm_calls_reach_tools_in_agent_loop(tmp_path: Path) -> None:
    inner = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    "inner",
                    usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                ),
            ],
        ),
    )

    @tool
    def probe() -> str:
        """Return the active workspace name after making a nested LLM call."""
        inner.invoke("hello")
        return get_settings().workspace_dir.name

    outer = _ToolCallingFake(
        messages=iter(
            [
                AIMessage(
                    "",
                    tool_calls=[{"name": "probe", "args": {}, "id": "call-1"}],
                    usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                ),
                AIMessage(
                    "done",
                    usage_metadata={"input_tokens": 150, "output_tokens": 5, "total_tokens": 155},
                ),
            ],
        ),
    )
    agent = create_agent(model=outer, tools=[probe])
    meter = RunMeter()
    with use_settings(_settings(tmp_path / "marker-ws")):
        config: RunnableConfig = {"callbacks": [meter]}
        result = agent.invoke({"messages": [HumanMessage("go")]}, config=config)

    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.content == "marker-ws"
    cost = meter.snapshot()
    assert cost.tool_calls == 1
    assert (cost.agent_llm_calls, cost.tool_llm_calls) == (2, 1)
    assert (cost.agent_input_tokens, cost.agent_output_tokens) == (250, 25)
    assert (cost.tool_input_tokens, cost.tool_output_tokens) == (7, 3)
    assert cost.total_tokens == 285


def test_level3_review_passes_outcome_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    papers = _papers()
    monkeypatch.setattr(reports, "search_arxiv", lambda *args, **kwargs: list(papers.values()))
    monkeypatch.setattr(
        reports,
        "summarize_paper",
        lambda arxiv_id, **kwargs: PaperSummary(
            arxiv_id=arxiv_id,
            summary_path="x.md",
            summary="A concise summary that is clearly long enough to count as one.",
            chunks_summarized=1,
            chunks_total=1,
        ),
    )
    settings = _settings(tmp_path)
    result = reports.conduct_review(
        "demo topic",
        settings=settings,
        llm=GenericFakeChatModel(messages=iter([])),
    )
    assert result.review_path == "reviews/review.md"
    score = evaluate_review(
        tmp_path / result.review_path,
        scenario=SCENARIO,
        expected_papers=5,
        lookup=_lookup(papers),
    )
    assert score.success, [c for c in score.criteria if not c.passed]


# --- runner ---------------------------------------------------------------------


def _patch_agent(monkeypatch: pytest.MonkeyPatch, behaviour: Callable[[], None]) -> None:
    monkeypatch.setattr(runner, "build_agent", lambda level, *, settings=None: object())
    monkeypatch.setattr(runner, "run_task", lambda agent, task, **kwargs: behaviour())


def _run(tmp_path: Path, *, repeats: int = 1, lookup: Any = None) -> tuple[int, Path]:
    output = tmp_path / "results.jsonl"
    executed = run_evaluation(
        scenarios=[SCENARIO],
        levels=["level1", "level3"],
        repeats=repeats,
        output=output,
        runs_dir=tmp_path / "runs",
        settings=_settings(tmp_path / "unused"),
        lookup=lookup or _lookup(_papers()),
    )
    return executed, output


def test_runner_records_isolated_runs_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write_review() -> None:
        workspace = get_settings().workspace_dir
        assert workspace.parent == tmp_path / "runs"
        assert get_settings().arxiv_date_from == SCENARIO.date_from
        _write_review(workspace, _review_text())

    _patch_agent(monkeypatch, write_review)
    executed, output = _run(tmp_path)
    assert executed == 2

    records = load_records(output)
    assert [r.run_id for r in records] == ["level1-demo-01", "level3-demo-01"]
    assert all(r.status == "completed" and r.outcome and r.outcome.success for r in records)
    assert records[0].temperature == 0.7
    assert len(records[0].system_prompt_sha256) == 64
    assert len(records[0].task_prompt_sha256) == 64

    again, _ = _run(tmp_path)
    assert again == 0
    more, _ = _run(tmp_path, repeats=2)
    assert more == 2
    assert len(load_records(output)) == 4


def test_runner_maps_step_limit_and_infrastructure_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hit_limit() -> None:
        raise GraphRecursionError

    _patch_agent(monkeypatch, hit_limit)
    _, output = _run(tmp_path)
    limited = load_records(output)
    assert {r.status for r in limited} == {"step_limit"}
    assert all(r.outcome and not r.outcome.success for r in limited)

    def lm_down() -> None:
        raise ConnectionError("LM Studio unreachable")

    error_output = tmp_path / "errors.jsonl"
    _patch_agent(monkeypatch, lm_down)
    run_evaluation(
        scenarios=[SCENARIO],
        levels=["level1"],
        repeats=1,
        output=error_output,
        runs_dir=tmp_path / "runs",
        settings=_settings(tmp_path / "unused"),
        lookup=_lookup({}),
    )
    [failed] = load_records(error_output)
    assert failed.status == "error"
    assert failed.outcome is None
    assert "ConnectionError" in (failed.error or "")

    # Error runs are retried on resume, not treated as finished.
    _patch_agent(monkeypatch, lambda: _write_review(get_settings().workspace_dir, _review_text()))
    run_evaluation(
        scenarios=[SCENARIO],
        levels=["level1"],
        repeats=1,
        output=error_output,
        runs_dir=tmp_path / "runs",
        settings=_settings(tmp_path / "unused"),
        lookup=_lookup(_papers()),
    )
    assert [r.status for r in load_records(error_output)] == ["error", "completed"]


def test_arxiv_failure_during_checks_is_an_error_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_agent(monkeypatch, lambda: _write_review(get_settings().workspace_dir, _review_text()))

    def broken_lookup(ids: Sequence[str]) -> Mapping[str, Any]:
        raise ExternalServiceError("arXiv down")

    _, output = _run(tmp_path, lookup=broken_lookup)
    assert {r.status for r in load_records(output)} == {"error"}


# --- report ---------------------------------------------------------------------


def _record(
    level: str,
    scenario: str,
    repeat: int,
    *,
    score: float,
    tokens: int,
    status: RunStatus = "completed",
) -> RunRecord:
    outcome = None
    if status != "error":
        outcome = OutcomeScore(
            criteria=[CriterionResult(name="x", passed=score == 1.0)],
            score=score,
            success=score == 1.0,
        )
    return RunRecord(
        run_id=f"{level}-{scenario}-{repeat:02d}",
        level=level,
        scenario_id=scenario,
        repeat=repeat,
        status=status,
        error="boom" if status == "error" else None,
        outcome=outcome,
        cost=CostMetrics(agent_input_tokens=tokens, tool_calls=2, agent_llm_calls=3),
        model="m",
        temperature=0.7,
        max_steps=10,
        prompt_version="v1",
        system_prompt_sha256="0" * 64,
        task_prompt_sha256="1" * 64,
        date_from=date(2024, 1, 1),
        date_to=date(2024, 6, 30),
        started_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def test_summary_statistics_and_error_exclusion() -> None:
    records = [
        _record("level1", "a", 1, score=1.0, tokens=100),
        _record("level1", "a", 2, score=0.5, tokens=300),
        _record("level1", "b", 1, score=1.0, tokens=200),
        _record("level1", "b", 2, score=1.0, tokens=200),
        _record("level1", "b", 3, score=0.0, tokens=0, status="error"),
    ]
    [summary] = summarize(records)
    assert (summary.runs, summary.error_runs) == (4, 1)
    assert summary.success_rate == pytest.approx(0.75)
    assert summary.mean_score == pytest.approx(0.875)
    assert summary.mean_total_tokens == pytest.approx(200)
    # Repeatability: std(a)=0.3536, std(b)=0 -> mean 0.1768.
    assert summary.repeatability_score_std == pytest.approx(0.1768, abs=1e-3)
    # Consistency: scenario means 0.75 and 1.0 -> sample std 0.1768.
    assert summary.consistency_score_std == pytest.approx(0.1768, abs=1e-3)
    assert summary.consistency_tokens_cv == 0.0


def test_report_renders_levels_scenarios_and_errors(tmp_path: Path) -> None:
    records = [
        _record("level1", "a", 1, score=1.0, tokens=100),
        _record("level2", "a", 1, score=0.5, tokens=50),
        _record("level2", "a", 2, score=0.0, tokens=0, status="error"),
    ]
    text = render_markdown(records)
    assert "level1" in text
    assert "level2" in text
    assert "level2-a-02" in text
    assert "boom" in text


def test_cli_report_prints_and_rejects_missing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = tmp_path / "r.jsonl"
    results.write_text(
        _record("level1", "a", 1, score=1.0, tokens=100).model_dump_json() + "\n",
        encoding="utf-8",
    )
    assert main(["report", str(results)]) == 0
    assert "Evaluation report" in capsys.readouterr().out
    assert main(["report", str(tmp_path / "missing.jsonl")]) == 2


# --- level parity, failure classification, config guard -----------------------------


def test_every_level_can_search_summarize_and_write() -> None:
    required = {
        "level1": {"search_arxiv", "summarize_chunk", "append_to_file"},
        "level2": {"search_papers", "get_and_summarize_paper", "append_to_review"},
        "level3": {"conduct_literature_review"},
    }
    for level, names in required.items():
        assert names <= {t.name for t in get_tools(level)}


def test_level2_building_blocks_produce_a_passing_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    papers = _papers()
    monkeypatch.setattr(reports, "search_arxiv", lambda *args, **kwargs: list(papers.values()))
    monkeypatch.setattr(reports, "fetch_arxiv_papers", lambda ids, **kwargs: list(papers.values()))
    settings = _settings(tmp_path)
    cited = reports.search_cited_papers("demo topic", max_results=5, settings=settings)
    assert cited[0].apa_citation.endswith("https://arxiv.org/abs/2401.00000")

    reports.append_to_review("# Review: demo topic", settings=settings)
    for number, paper in enumerate(cited, start=1):
        reports.append_to_review(
            f"## {number}. {paper.title}\n\n{paper.apa_citation}\n\n"
            "This paper studies the topic and reports clear, reproducible results.",
            settings=settings,
        )
    reports.append_bibliography([p.arxiv_id for p in cited], settings=settings)
    score = evaluate_review(
        tmp_path / "reviews" / "review.md",
        scenario=SCENARIO,
        expected_papers=5,
        lookup=_lookup(papers),
    )
    assert score.success, [c for c in score.criteria if not c.passed]


def test_split_keeps_only_leading_chunks(tmp_path: Path) -> None:
    settings = _settings(tmp_path, chunk_size=500, pipeline_max_chunks=2)
    text = tmp_path / "texts" / "doc.txt"
    text.parent.mkdir()
    text.write_text("word " * 1000, encoding="utf-8")
    assert len(split_paper_text("texts/doc.txt", settings=settings)) == 2


def test_context_overflow_is_a_scored_failure_not_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def overflow() -> None:
        response = httpx2.Response(400, request=httpx2.Request("POST", "http://llm/v1"))
        message = "Context size has been exceeded."
        raise openai.BadRequestError(message, response=response, body=None)

    _patch_agent(monkeypatch, overflow)
    _, output = _run(tmp_path)
    records = load_records(output)
    assert {r.status for r in records} == {"failed"}
    assert all(r.outcome and not r.outcome.success for r in records)
    assert {r.run_id for r in records} == {"level1-demo-01", "level3-demo-01"}
    # Failed runs are final: a second invocation does not repeat them.
    assert _run(tmp_path)[0] == 0
    [level1, _] = summarize(records)
    assert (level1.runs, level1.failed_runs, level1.error_runs) == (1, 1, 0)
    assert level1.success_rate == 0.0


def test_resume_rejects_output_from_another_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_agent(monkeypatch, lambda: _write_review(get_settings().workspace_dir, _review_text()))
    _, output = _run(tmp_path)
    with pytest.raises(InvalidInputError, match="different configuration"):
        run_evaluation(
            scenarios=[SCENARIO],
            levels=["level1"],
            repeats=2,
            output=output,
            runs_dir=tmp_path / "runs",
            settings=_settings(tmp_path / "unused", chunk_size=1000),
            lookup=_lookup(_papers()),
        )


def test_repeatability_skips_scenarios_with_a_single_run() -> None:
    records = [
        _record("level1", "a", 1, score=1.0, tokens=100),
        _record("level1", "b", 1, score=0.5, tokens=100),
        _record("level1", "b", 2, score=0.0, tokens=100),
    ]
    [summary] = summarize(records)
    assert summary.repeatability_score_std == pytest.approx(0.3536, abs=1e-3)
