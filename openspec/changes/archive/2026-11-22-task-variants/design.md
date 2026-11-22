# Design

## Context

The motivation is in proposal.md. The current state:

- `Scenario` (`evaluation/models.py`) has `id`, `topic`, `date_from` and `date_to`. `scenarios/scenarios.json` holds 10 base scenarios, and `load_scenarios` validates that ids are unique.
- `runner.run_single` renders `load_prompt("evaluation", version)` with `topic`, `n_papers`, `date_from` and `date_to`. `config_fingerprint` hashes the system and task prompts.
- Run ids are `<level>-<scenario>-<repeat>`. Blocks are planned per (repeat, scenario) by `block_order`.
- `report.py` groups records by model and level and pairs levels by `scenario_id`.
- After `protocol-fixes`, prompts are at version v3 and describe only the outcome.

## Goals / Non-Goals

**Goals**
- Variants exactly as described in `data_environment.tex`: three requirements, assigned in turn, 4/3/3.
- No change to tools, levels, ordering or resume logic beyond what a new scenario needs.
- Base and variant results are always kept apart in the report.

**Non-Goals**
- Checking whether a requirement was met. That is the `requirement_met` check in `outcome-evaluation`: deterministic for `table` and `polish`, by the judge for `no_surveys`.
- Requirement-specific tools, or parameters on the level-3 tool. Level 3 stays fixed by design.
- Statistical testing of H1b. That belongs to the analysis plan in `outcome-evaluation`.

## Decisions

### Variants are ordinary scenarios with two extra fields
- `Scenario` gains `requirement: Literal["table", "polish", "no_surveys"] | None = None` and `base: str | None = None`.
- A variant has both fields set; a base scenario has neither.
- `load_scenarios` checks three things:
  - each variant's `base` exists and is itself a base scenario;
  - its topic and window equal the base's;
  - its id is `<base>-<requirement>`.
- Run ids, blocks, ordering, resume and `--only` then work unchanged.
- *Alternative:* a task-type dimension in the run plan (base × variant per block). Rejected, because it would change run ids, block planning and resume for no gain.

### Variants are written into the scenarios file, not generated
The 10 variant entries are written explicitly into `scenarios/scenarios.json`, in rotation `table`, `polish`, `no_surveys`, … over the base scenarios in file order:

| Requirement | Variants |
|---|---|
| `table` | `rag-2024h1`, `rlhf-2024`, `protein-2022h2`, `xai-2022` |
| `polish` | `gnn-2023h1`, `federated-2022`, `timeseries-2023` |
| `no_surveys` | `diffusion-2023h2`, `qec-2023`, `hallucination-2024` |

A test checks that the file follows the rotation. An explicit file keeps the scenario set readable and versioned.

### Requirement text in versioned prompt files
- New prompts:
  - `prompts/requirement_table/v3.md`: "Additionally, place directly under the first-level heading a Markdown table comparing the selected papers, with one row per paper (columns: title, year, main contribution)."
  - `prompts/requirement_polish/v3.md`: "Additionally, write every paper summary in Polish; titles and references stay as on arXiv."
  - `prompts/requirement_no_surveys/v3.md`: "Additionally, do not include survey or review papers; every cited paper must present original work."
- The runner appends the rendered requirement as a final line of the task prompt.
- The fingerprint gains the sha256 of each requirement prompt, hashed for every run because the set of prompts is part of the protocol.
- The wording of each requirement is chosen so that the check in `outcome-evaluation` can verify it. That is why `table` asks for a table with one row per paper, and `polish` says that titles and references stay as on arXiv.

### Record fields
- `RunRecord` gains `task_type: Literal["base", "variant"] = "base"`, `requirement: str | None = None` and `base_scenario_id: str | None = None`. The defaults keep old files loading as base tasks.

### Report
- `report.py` partitions records by task type before grouping.
- Every existing section is rendered once for base tasks. For variants it adds the same quality and cost tables, plus success by requirement.
- `compare_levels` pairs scenarios within a task type.
- The CSV gains a `task_type` column. Rows are per model, level and task type.

## Risks / Trade-offs

- **[The requirements favour levels that can compose.]** That is the hypothesis under test. The paper already lists this as a limitation, and the results are reported separately.
- **[Level 3 can rarely meet a requirement except by chance.]** For example, `no_surveys` passes when no survey happens to be among the top papers. This is expected, and the check result reports it per requirement.
- **[Polish summaries break base checks.]** `titles_match` and `citations_apa` look at the title and the reference, which stay as on arXiv. `summaries_present` counts characters in any language. No base check depends on the language of the summary.
- **[The final run doubles to 900 runs.]** Accepted by the group. The time estimate comes from the `protocol-fixes` smoke run.

## Migration Plan

1. Implement after `protocol-fixes`.
2. Run a smoke test on the Mac with qwen-mid on 3 variants (one per requirement), all levels.
3. The final run uses the full 20-scenario file.

Rollback: remove the variant entries from the scenarios file. The code stays compatible with base-only files.

## Open Questions

None.
