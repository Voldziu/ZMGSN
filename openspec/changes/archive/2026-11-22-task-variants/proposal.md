# Proposal

## Why

H1b says that, on tasks with an added requirement, level 2 succeeds more often than level 3, because a fixed macro-tool cannot adapt. Together with H1a, it places level 2 on the success–cost Pareto front. The paper describes these variant tasks in `data_environment.tex`, but there is no code or scenario for them, so the final run could not test H1b (methodology review, 2026-11-21). The group decided to implement the variants before the final run and not drop H1b.

## What Changes

- **Variant scenarios.** Every base scenario gets one variant with the same topic and date window plus one added requirement. The three requirements from `data_environment.tex` are assigned in turn over the scenarios in file order, so they cover 4, 3 and 3 scenarios:
  - `table`: a comparison table of the selected papers, placed under the main heading and before the first paper section;
  - `polish`: every summary written in Polish;
  - `no_surveys`: no survey or review paper may be cited.

  Variants are scenarios of their own, with an id `<base-id>-<requirement>` and a reference to their base scenario. They are listed in `scenarios/scenarios.json`, which then holds 20 scenarios.
- **Variant task prompt.** The task prompt of a variant is the base prompt with one extra line that states the requirement. The line comes from a versioned prompt file for each requirement, and its hash enters the configuration hash. Base tasks get exactly the base prompt.
- **Tools stay the same.** Every level keeps its tools, and no level gets a requirement-specific tool. Level 1 already has `translate_text` and `create_markdown_table`. Level 2 must write the table or the Polish text itself. Level 3 cannot adapt, and that is the effect H1b measures.
- **Run records.** Each record carries the task type (`base` or `variant`), the requirement and the base scenario id.
- **Report.** Base tasks and variants are summarised separately and never pooled. Level differences are computed within a task type; H1b is the variant comparison of level 2 against level 3.
- **Check of the requirement.** The added check for each requirement (table and Polish deterministic, `no_surveys` through the LLM judge) belongs to the `outcome-evaluation` change. Until it lands, variant runs are scored with the base checks only.
- **Plan and paper.**
  - The final run covers 20 scenarios × 3 levels × 3 repeats per model, which is 900 runs over 5 models.
  - `data_environment.tex` and `experiment_plan.tex` describe the implemented variants and the run counts. The statement that the variants are "implemented in the next stage" is removed.

## Capabilities

### New Capabilities
- `task-variants`: what a variant task is, how variants are derived from base scenarios, how their prompt is built, and how runs record them.

### Modified Capabilities
- `evaluation-report`: base and variant tasks are reported separately, and level differences are computed within a task type.

## Impact

- **Code:**
  - `evaluation/models.py`: `Scenario` gains `requirement` and `base`; `RunRecord` gains `task_type`, `requirement` and `base_scenario_id`.
  - `evaluation/scenarios.py`: validation that each variant points to an existing base scenario.
  - `evaluation/runner.py`: prompt rendering and the fingerprint.
  - New prompt files `prompts/requirement_{table,polish,no_surveys}/v3.md`.
  - `evaluation/report.py`: separate sections by task type.
  - `scenarios/scenarios.json`.
- **Run time:** the final run doubles, to 900 runs.
- **Dependencies:** this change builds on `protocol-fixes` (prompt v3). Their order is protocol-fixes first, then task-variants.
- **Results:** the new scenarios and prompt files change the hash. Base runs keep their ids, so a base-only run and a run with variants are comparable when both use the same hash.
- **Docs:** `README.md`, `docs/overleaf/data_environment.tex`, `docs/overleaf/experiment_plan.tex`.
