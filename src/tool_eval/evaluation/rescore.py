"""Recompute the deterministic checks of saved runs under the current scoring rules.

`rescore` reads a results file and, for every scored run whose workspace still exists, runs
`evaluate_review` again: format checks, fact checks and the `table`/`polish` requirement checks.
It writes a new results file and leaves the original untouched; runs without a workspace keep
their old outcome and are listed. The configuration hash of the records is kept, so rescored
records still identify the run protocol they were made with.
"""

from collections.abc import Mapping
from pathlib import Path

from tool_eval.evaluation.checks import Lookup, evaluate_review
from tool_eval.evaluation.models import RunRecord, Scenario
from tool_eval.services.reports import REVIEW_PATH

RESCORED_SUFFIX = ".rescored.jsonl"


def rescore(
    results: Path,
    *,
    runs_dir: Path,
    scenarios: Mapping[str, Scenario],
    lookup: Lookup,
    output: Path | None = None,
) -> tuple[Path, list[str]]:
    """Write a rescored copy of a results file.

    Args:
        results: Results file to rescore.
        runs_dir: Directory with one workspace per run.
        scenarios: Scenarios by id (window and requirement).
        lookup: arXiv metadata by id.
        output: Destination; defaults to `<results stem>.rescored.jsonl` next to it.

    Returns:
        The written file, and the runs whose workspace or scenario was missing.
    """
    target = output or results.with_name(results.stem + RESCORED_SUFFIX)
    missing: list[str] = []
    lines = []
    for line in results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = RunRecord.model_validate_json(line)
        workspace = runs_dir / record.run_id
        scenario = scenarios.get(record.scenario_id)
        if record.status != "error" and (scenario is None or not workspace.is_dir()):
            missing.append(record.run_id)
        elif record.status != "error" and scenario is not None:
            outcome = evaluate_review(
                workspace / REVIEW_PATH,
                workspace=workspace,
                scenario=scenario,
                expected_papers=record.review_paper_count or 5,
                lookup=lookup,
            )
            record = record.model_copy(update={"outcome": outcome})
        lines.append(record.model_dump_json())
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target, missing
