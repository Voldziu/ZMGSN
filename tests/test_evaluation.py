import json
import re
from collections import defaultdict
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

import tool_eval.main as cli
from tool_eval.config import Settings, get_settings, use_settings
from tool_eval.evaluation import runner
from tool_eval.evaluation.checks import CHECKS_VERSION, evaluate_review
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
from tool_eval.llm import get_chat_model
from tool_eval.main import main
from tool_eval.prompts import load_prompt
from tool_eval.services import reports
from tool_eval.services.arxiv import ArxivPaper
from tool_eval.services.papers import PaperSummary, split_paper_text
from tool_eval.tools.registry import TOOLSET_VERSION, TOOLSETS, get_tools

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


def _write_review(workspace: Path, text: str, *, read: bool = True) -> None:
    target = workspace / "reviews" / "review.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    if read:
        _mark_read(workspace, re.findall(r"arxiv\.org/abs/(\S+)", text))


def _mark_read(workspace: Path, arxiv_ids: Sequence[str]) -> None:
    """Leave the extracted full text the tools would have written for each paper."""
    for arxiv_id in arxiv_ids:
        target = workspace / "texts" / f"{arxiv_id}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("full text", encoding="utf-8")


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
        workspace=tmp_path,
        scenario=SCENARIO,
        expected_papers=5,
        lookup=_lookup(_papers()),
    )
    assert score.success, [c for c in score.criteria if not c.passed]
    assert score.score == 1.0


def test_missing_file_fails_every_criterion(tmp_path: Path) -> None:
    score = evaluate_review(
        tmp_path / "reviews/review.md",
        workspace=tmp_path,
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
        workspace=tmp_path,
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
        workspace=tmp_path,
        scenario=SCENARIO,
        expected_papers=5,
        lookup=_lookup({}),
    )
    assert "ids_exist" in {c.name for c in score.criteria if not c.passed}


def _papers_read(tmp_path: Path) -> CriterionResult:
    score = evaluate_review(
        tmp_path / "reviews/review.md",
        workspace=tmp_path,
        scenario=SCENARIO,
        expected_papers=5,
        lookup=_lookup(_papers()),
    )
    return next(c for c in score.criteria if c.name == "papers_read")


def test_review_from_search_results_only_fails_papers_read(tmp_path: Path) -> None:
    _write_review(tmp_path, _review_text(), read=False)
    criterion = _papers_read(tmp_path)
    assert not criterion.passed
    assert all(f"2401.0000{i}" in criterion.detail for i in range(5))


def test_one_unread_paper_fails_papers_read(tmp_path: Path) -> None:
    _write_review(tmp_path, _review_text(), read=False)
    _mark_read(tmp_path, [f"2401.0000{i}" for i in range(4)])
    criterion = _papers_read(tmp_path)
    assert not criterion.passed
    assert "2401.00004" in criterion.detail
    assert "2401.00000" not in criterion.detail


def test_all_papers_read_passes(tmp_path: Path) -> None:
    _write_review(tmp_path, _review_text())
    assert _papers_read(tmp_path).passed


def test_pdf_in_shared_cache_only_does_not_count(tmp_path: Path) -> None:
    _write_review(tmp_path, _review_text(), read=False)
    cache = tmp_path.parent / "pdf_cache"
    cache.mkdir(exist_ok=True)
    for i in range(5):
        (cache / f"2401.0000{i}.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "papers").mkdir(exist_ok=True)
        (tmp_path / "papers" / f"2401.0000{i}.pdf").write_bytes(b"%PDF-1.4")
    assert not _papers_read(tmp_path).passed


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
    meter = RunMeter(tool_names={"probe"})
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
    assert cost.first_call_input_tokens == 100
    assert cost.malformed_tool_calls == 0
    assert cost.agent_llm_seconds is not None
    assert cost.tool_llm_seconds is not None
    assert cost.tool_seconds is not None
    assert min(cost.agent_llm_seconds, cost.tool_llm_seconds) >= 0
    assert cost.tool_seconds >= cost.tool_llm_seconds
    assert cost.wall_seconds >= cost.agent_llm_seconds


def test_level3_review_passes_outcome_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    papers = _papers()
    monkeypatch.setattr(reports, "search_arxiv", lambda *args, **kwargs: list(papers.values()))

    def fake_summarize(arxiv_id: str, **kwargs: Any) -> PaperSummary:
        _mark_read(tmp_path, [arxiv_id])  # the real tool extracts the text first
        return PaperSummary(
            arxiv_id=arxiv_id,
            summary_path="x.md",
            summary="A concise summary that is clearly long enough to count as one.",
            chunks_summarized=1,
            chunks_total=1,
        )

    monkeypatch.setattr(reports, "summarize_paper", fake_summarize)
    settings = _settings(tmp_path)
    result = reports.conduct_review(
        "demo topic",
        settings=settings,
        llm=GenericFakeChatModel(messages=iter([])),
    )
    assert result.review_path == "reviews/review.md"
    score = evaluate_review(
        tmp_path / result.review_path,
        workspace=tmp_path,
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
    assert sorted(r.run_id for r in records) == ["level1-demo-01", "level3-demo-01"]
    assert all(r.status == "completed" and r.success for r in records)
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

    def server_down() -> None:
        raise ConnectionError("vLLM server unreachable")

    error_output = tmp_path / "errors.jsonl"
    _patch_agent(monkeypatch, server_down)
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


LEVELS = ["level1", "level2", "level3"]
# block_order(LEVELS, seed=7, repeat=1, scenario_id="demo"), fixed when the order was introduced.
GOLDEN_ORDER = ["level1", "level3", "level2"]


def test_block_order_is_a_reproducible_permutation() -> None:
    order = runner.block_order(LEVELS, seed=7, repeat=1, scenario_id="demo")
    assert sorted(order) == LEVELS
    # Only the set of levels matters, not how they were listed.
    assert runner.block_order(LEVELS[::-1], seed=7, repeat=1, scenario_id="demo") == order
    assert runner.block_order([*LEVELS, "level1"], seed=7, repeat=1, scenario_id="demo") == order
    # A recorded seed must keep reproducing the orders it planned.
    assert order == GOLDEN_ORDER
    orders = {
        tuple(runner.block_order(LEVELS, seed=7, repeat=1, scenario_id=f"s{i}")) for i in range(10)
    }
    assert len(orders) > 1


def _run_levels(
    tmp_path: Path,
    *,
    repeats: int,
    order_seed: int | None,
    name: str = "results.jsonl",
    levels: Sequence[str] = tuple(LEVELS),
) -> list[RunRecord]:
    output = tmp_path / name
    run_evaluation(
        scenarios=[SCENARIO, SCENARIO.model_copy(update={"id": "demo2"})],
        levels=levels,
        repeats=repeats,
        output=output,
        runs_dir=tmp_path / "runs",
        settings=_settings(tmp_path / "unused"),
        lookup=_lookup(_papers()),
        order_seed=order_seed,
    )
    return load_records(output)


def _blocks(records: Sequence[RunRecord]) -> dict[tuple[int, str], list[RunRecord]]:
    blocks: dict[tuple[int, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        blocks[record.repeat, record.scenario_id].append(record)
    return blocks


def _assert_block_order(records: Sequence[RunRecord], seed: int) -> None:
    for (repeat, scenario_id), block in _blocks(records).items():
        expected = runner.block_order(LEVELS, seed=seed, repeat=repeat, scenario_id=scenario_id)
        assert [r.level for r in block] == expected
        assert [r.block_position for r in block] == [1, 2, 3]


def test_levels_run_in_the_recorded_block_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_agent(monkeypatch, lambda: _write_review(get_settings().workspace_dir, _review_text()))
    records = _run_levels(tmp_path, repeats=2, order_seed=7)
    assert len(records) == 12
    assert {r.order_seed for r in records} == {7}
    # Repeat 1 finishes everywhere before repeat 2 starts.
    assert [r.repeat for r in records] == [1] * 6 + [2] * 6
    _assert_block_order(records, seed=7)


def test_resumed_evaluation_keeps_its_order_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_agent(monkeypatch, lambda: _write_review(get_settings().workspace_dir, _review_text()))
    _run_levels(tmp_path, repeats=1, order_seed=7)
    records = _run_levels(tmp_path, repeats=2, order_seed=None)
    assert len(records) == 12
    assert {r.order_seed for r in records} == {7}
    _assert_block_order(records, seed=7)


def test_new_results_file_draws_an_order_seed_outside_the_config_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_agent(monkeypatch, lambda: _write_review(get_settings().workspace_dir, _review_text()))
    drawn = _run_levels(tmp_path, repeats=1, order_seed=None, name="a.jsonl")
    [seed] = {r.order_seed for r in drawn}
    assert isinstance(seed, int)
    _assert_block_order(drawn, seed=seed)
    other = _run_levels(tmp_path, repeats=1, order_seed=seed + 1, name="b.jsonl")
    assert {r.config_sha256 for r in drawn} == {r.config_sha256 for r in other}


def test_resuming_with_more_levels_keeps_positions_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_agent(monkeypatch, lambda: _write_review(get_settings().workspace_dir, _review_text()))
    _run_levels(tmp_path, repeats=1, order_seed=7, levels=["level1"])
    records = _run_levels(tmp_path, repeats=1, order_seed=None)
    assert len(records) == 6
    for (repeat, scenario_id), block in _blocks(records).items():
        plan = runner.block_order(list(TOOLSETS), seed=7, repeat=repeat, scenario_id=scenario_id)
        assert sorted(r.block_position or 0 for r in block) == [1, 2, 3]
        assert all(plan[(r.block_position or 0) - 1] == r.level for r in block)


def test_a_subset_of_levels_keeps_its_planned_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_agent(monkeypatch, lambda: _write_review(get_settings().workspace_dir, _review_text()))
    records = _run_levels(tmp_path, repeats=2, order_seed=7, levels=["level3", "level1"])
    assert len(records) == 8
    for (repeat, scenario_id), block in _blocks(records).items():
        plan = runner.block_order(list(TOOLSETS), seed=7, repeat=repeat, scenario_id=scenario_id)
        assert [r.level for r in block] == [level for level in plan if level != "level2"]
        assert [r.block_position for r in block] == [plan.index(r.level) + 1 for r in block]


def test_retried_run_keeps_its_planned_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = {"up_for_level2": False}

    def run_task(agent: Any, task: str, *, metadata: Mapping[str, Any], **kwargs: Any) -> list[Any]:
        if metadata["level"] == "level2" and not server["up_for_level2"]:
            raise ConnectionError("vLLM server unreachable")
        _write_review(get_settings().workspace_dir, _review_text())
        return []

    monkeypatch.setattr(runner, "build_agent", lambda level, *, settings=None: object())
    monkeypatch.setattr(runner, "run_task", run_task)
    _run_levels(tmp_path, repeats=1, order_seed=7)
    server["up_for_level2"] = True
    records = _run_levels(tmp_path, repeats=1, order_seed=None)

    errors = [r for r in records if r.status == "error"]
    assert len(errors) == 2
    for error in errors:
        block = _blocks(records)[error.repeat, error.scenario_id]
        [retry] = [r for r in block if r.level == "level2" and r.status == "completed"]
        assert retry.block_position == error.block_position
        # The start time shows that the retry ran after the rest of its block.
        assert retry.started_at == max(r.started_at for r in block)


def test_cli_passes_the_order_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    seen: dict[str, Any] = {}

    def fake_run_evaluation(**kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_evaluation", fake_run_evaluation)
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(
        '[{"id": "demo", "topic": "demo topic", "date_from": "2024-01-01",'
        ' "date_to": "2024-06-30"}]',
        encoding="utf-8",
    )
    assert main(["run", "--scenarios", str(scenarios), "--order-seed", "7"]) == 0
    assert seen["order_seed"] == 7
    assert main(["run", "--scenarios", str(scenarios)]) == 0
    assert seen["order_seed"] is None


# --- success rule ---------------------------------------------------------------


def _outcome(passed: int, total: int = 10) -> OutcomeScore:
    return OutcomeScore.from_criteria(
        [CriterionResult(name=f"check{i}", passed=i < passed) for i in range(total)]
    )


@pytest.mark.parametrize(
    ("status", "passed", "success", "score"),
    [
        ("completed", 10, True, 1.0),
        ("completed", 9, False, 0.9),
        ("step_limit", 10, False, 1.0),
        ("failed", 6, False, 0.6),
    ],
)
def test_only_completed_runs_that_pass_every_check_succeed(
    status: RunStatus, passed: int, success: bool, score: float
) -> None:
    record = _record("level1", "a", 1, score=0.0, tokens=0).model_copy(
        update={"status": status, "outcome": _outcome(passed)}
    )
    assert record.success is success
    assert record.outcome is not None
    assert record.outcome.score == pytest.approx(score)


def test_success_is_written_but_recomputed_on_read() -> None:
    record = _record("level1", "a", 1, score=1.0, tokens=10)
    assert json.loads(record.model_dump_json())["success"] is True

    # A line written before the rule: no top-level `success`, every check passed.
    limited = _record("level1", "a", 1, score=1.0, tokens=10, status="step_limit")
    old = json.loads(limited.model_dump_json())
    del old["success"]
    assert old["outcome"]["success"] is True
    assert RunRecord.model_validate(old).success is False
    # A stored value never overrides the rule.
    assert RunRecord.model_validate({**old, "success": True}).success is False


def test_step_limit_after_a_complete_review_is_not_a_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def review_then_loop() -> None:
        _write_review(get_settings().workspace_dir, _review_text())
        raise GraphRecursionError

    _patch_agent(monkeypatch, review_then_loop)
    _, output = _run(tmp_path)
    records = load_records(output)
    assert {r.status for r in records} == {"step_limit"}
    for record in records:
        assert record.outcome is not None
        assert record.outcome.success
        assert record.outcome.score == 1.0
        assert not record.success
    assert all(not json.loads(line)["success"] for line in output.read_text().splitlines())
    [level1, _] = summarize(records)
    assert level1.success_rate == 0.0
    assert level1.mean_score == 1.0


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
        "level2": {"search_papers", "get_and_summarize_paper", "write_review"},
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

    sections = [
        f"## {number}. {paper.title}\n\n{paper.apa_citation}\n\n"
        "This paper studies the topic and reports clear, reproducible results."
        for number, paper in enumerate(cited, start=1)
    ]
    reports.write_review("\n\n".join(["# Review: demo topic", *sections]), settings=settings)
    reports.append_bibliography([p.arxiv_id for p in cited], settings=settings)
    _mark_read(tmp_path, [p.arxiv_id for p in cited])
    score = evaluate_review(
        tmp_path / "reviews" / "review.md",
        workspace=tmp_path,
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


def test_fingerprint_covers_step_limit_and_parallel_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("AGENT_MAX_STEPS", "LLM_PARALLEL_TOOL_CALLS"):
        monkeypatch.delenv(name, raising=False)

    def fingerprint(**extra: Any) -> str:
        return runner.config_fingerprint(_settings(tmp_path, llm_model="qwen", **extra))

    base = fingerprint()
    assert fingerprint(agent_max_steps=10) != base
    assert fingerprint(llm_parallel_tool_calls=False) != base


def test_fingerprint_covers_toolset_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = runner.config_fingerprint(_settings(tmp_path))
    monkeypatch.setattr(runner, "TOOLSET_VERSION", TOOLSET_VERSION + 1)
    assert runner.config_fingerprint(_settings(tmp_path)) != base


def test_each_run_gets_its_own_recorded_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int | None] = []

    def behaviour() -> None:
        # A tool-internal model built inside the run sees the run's seed.
        seen.append(get_chat_model().seed)
        _write_review(get_settings().workspace_dir, _review_text())

    _patch_agent(monkeypatch, behaviour)
    _, output = _run(tmp_path, repeats=2)

    records = load_records(output)
    seeds = [r.seed for r in records]
    assert all(isinstance(s, int) for s in seeds)
    assert len(set(seeds)) == len(seeds)
    assert seen == seeds
    assert len({r.config_sha256 for r in records}) == 1


def test_runs_share_an_arxiv_cache_next_to_the_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path | None] = []

    def behaviour() -> None:
        seen.append(get_settings().arxiv_cache_dir)
        _write_review(get_settings().workspace_dir, _review_text())

    _patch_agent(monkeypatch, behaviour)
    _run(tmp_path)
    assert seen == [tmp_path / "arxiv_cache"] * 2


def test_fingerprint_covers_checks_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = runner.config_fingerprint(_settings(tmp_path))
    monkeypatch.setattr(runner, "CHECKS_VERSION", CHECKS_VERSION + 1)
    assert runner.config_fingerprint(_settings(tmp_path)) != base


def _final_reply(monkeypatch: pytest.MonkeyPatch, reply: AIMessage) -> None:
    def run_task(agent: Any, task: str, **kwargs: Any) -> list[Any]:
        _write_review(get_settings().workspace_dir, _review_text())
        return [HumanMessage(task), reply]

    monkeypatch.setattr(runner, "build_agent", lambda level, *, settings=None: object())
    monkeypatch.setattr(runner, "run_task", run_task)


def test_reply_cut_by_token_limit_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cut = AIMessage(
        '{"name": "write_review", "parameters": {"markdown": "# Review',
        response_metadata={"finish_reason": "length"},
    )
    _final_reply(monkeypatch, cut)
    _, output = _run(tmp_path)
    record = load_records(output)[0]
    assert record.status == "failed"
    assert record.error is not None
    assert "LLM_MAX_TOKENS" in record.error


def test_normal_final_reply_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _final_reply(
        monkeypatch,
        AIMessage("reviews/review.md", response_metadata={"finish_reason": "stop"}),
    )
    _, output = _run(tmp_path)
    assert load_records(output)[0].status == "completed"
