## 1. Scenarios

- [ ] 1.1 Add `requirement` and `base` to `Scenario`. In `load_scenarios`, validate variants: `base` exists and is a base scenario, the topic and window match, and the id is `<base>-<requirement>`. Tests for each failure and for a valid file.
- [ ] 1.2 Add the 10 variants to `scenarios/scenarios.json` in the `table` / `polish` / `no_surveys` rotation. Test: the file follows the rotation (4/3/3), and the first variant is `rag-2024h1-table`.

## 2. Prompts and runner

- [ ] 2.1 Add `prompts/requirement_table/v3.md`, `prompts/requirement_polish/v3.md` and `prompts/requirement_no_surveys/v3.md`.
- [ ] 2.2 In `run_single`, append the rendered requirement to the task prompt of variants. Test: the variant prompt equals the base prompt plus one requirement statement, and the base prompt is unchanged.
- [ ] 2.3 Add the sha256 of the requirement prompts to `config_fingerprint`. Test: editing a requirement prompt changes the hash.
- [ ] 2.4 Add `task_type`, `requirement` and `base_scenario_id` to `RunRecord`, with defaults for old records. Tests: a variant record carries the fields; an old record loads as `base`.
- [ ] 2.5 Test that `--only` with variant ids and block ordering work for variants unchanged.

## 3. Report

- [ ] 3.1 In `report.py`, split records by task type, render a variant section (the same tables, plus success by requirement), and pair level differences within a task type. Tests: mixed file, base-only file, and `level3 − level2` on variants only.
- [ ] 3.2 Add a `task_type` column to the CSV, with one row per model, level and task type. Update the CSV test.

## 4. Docs and paper

- [ ] 4.1 `README.md`: variant scenarios, requirement prompts, report sections.
- [ ] 4.2 `docs/overleaf/data_environment.tex`: describe the implemented variants (ids, rotation, requirement wording), and remove "implemented in the next stage".
- [ ] 4.3 `docs/overleaf/experiment_plan.tex`: 20 scenarios × 3 levels × 3 repeats per model (900 runs), and H1b tested as `level3 − level2` on variants.
- [ ] 4.4 `docs/presentations/src/progress-6-7.html`: add a slide on the variants and H1b to the plan slides, and regenerate the PDF.

## 5. Verification

- [ ] 5.1 `make test`, ruff and mypy clean.
- [ ] 5.2 Smoke run on the Mac: qwen-mid, all levels, `rag-2024h1-table`, `gnn-2023h1-polish`, `diffusion-2023h2-no_surveys`, REPEATS=1. Check the prompts in the trace, the record fields and the variant section of the report.
