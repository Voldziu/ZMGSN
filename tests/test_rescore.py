import json
from pathlib import Path

from test_evaluation import SCENARIO, _lookup, _mark_read, _papers, _record, _review_text
from tool_eval.evaluation.rescore import rescore


def test_rescoring_adds_facts_and_keeps_the_original(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    old = _record("level1", "demo", 1, score=0.1, tokens=1)
    gone = _record("level2", "demo", 1, score=1.0, tokens=1)
    workspace = runs / old.run_id
    review = workspace / "reviews" / "review.md"
    review.parent.mkdir(parents=True)
    review.write_text(_review_text().replace("## ", "### "), encoding="utf-8")
    _mark_read(workspace, [f"2401.0000{i}" for i in range(5)])
    results = tmp_path / "presets-m.jsonl"
    original = old.model_dump_json() + "\n" + gone.model_dump_json() + "\n"
    results.write_text(original, encoding="utf-8")

    written, missing = rescore(
        results, runs_dir=runs, scenarios={"demo": SCENARIO}, lookup=_lookup(_papers())
    )

    assert results.read_text(encoding="utf-8") == original
    assert written.name == "presets-m.rescored.jsonl"
    assert missing == [gone.run_id]
    first, second = (json.loads(line) for line in written.read_text().splitlines())
    assert all(f["passed"] for f in first["outcome"]["facts"])
    assert second["outcome"]["facts"] == []  # kept as it was
