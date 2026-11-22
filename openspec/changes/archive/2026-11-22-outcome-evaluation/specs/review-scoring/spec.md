# Spec Delta

## ADDED Requirements

### Requirement: Fact checks read ids from the whole file
Each run SHALL be scored with four fact checks. They SHALL take the cited papers from every arXiv identifier anywhere in the review file, whatever the headings, list markers or emphasis around it:
- `papers_cited`: at least the requested number of distinct papers are cited;
- `cited_ids_exist`: every cited id exists on arXiv;
- `cited_in_window`: every cited paper was submitted in the scenario's date window;
- `text_extracted`: every cited paper's text was extracted in the run's own workspace.

#### Scenario: Wrong heading level
- **GIVEN** a review that cites five real, in-window papers whose texts were extracted, with `###` paper headings
- **WHEN** the run is scored
- **THEN** all four fact checks pass, while the format check `section_count` fails

#### Scenario: Invented identifier
- **WHEN** a review cites `2401.00002`, which was never extracted in the run
- **THEN** `text_extracted` fails and names that id

### Requirement: Variants are scored on their requirement
Each variant run SHALL be scored with a `requirement_met` check for its requirement:
- `table`: a Markdown table between the first-level heading and the first paper section, with a row for every cited paper (by title or id);
- `polish`: the summary text of every paper section, without titles, references and URLs, is detected as Polish;
- `no_surveys`: the judge finds no survey among the cited papers; the check is undefined until the run is judged.

Base runs SHALL have no `requirement_met` check.

#### Scenario: Table variant at level 3
- **WHEN** a `table` variant run at level 3 produces the level-3 tool's fixed review, which has no table
- **THEN** `requirement_met` fails

#### Scenario: Polish summaries
- **WHEN** every section of a `polish` variant review holds an English title and reference followed by a Polish summary
- **THEN** `requirement_met` passes

### Requirement: Three outcomes per run
Each scored run SHALL have three outcomes:
- **artifact valid:** every fact check passes, every judge rating is at least 4, and a variant meets its requirement. It is undefined while a needed judgment is missing.
- **reliable:** the run ended with status `completed`.
- **success:** the run is both artifact valid and reliable.

The existing rule (`completed` and every format check passes) SHALL remain as **format success**.

#### Scenario: Loop after a valid review
- **GIVEN** a level-3 run whose review passes the fact checks and is rated 4 or 5 on every scale, and which ended as `loop`
- **WHEN** its outcomes are computed
- **THEN** it is artifact valid, not reliable, and not a success

#### Scenario: Valid content with wrong format
- **GIVEN** a completed run whose review passes the fact checks and the judge, but uses `###` headings
- **WHEN** its outcomes are computed
- **THEN** it is a success but not a format success

### Requirement: Saved runs can be rescored
A `rescore` command SHALL recompute the deterministic checks (format, facts, and the `table` and `polish` requirements) of every run in a results file from the run's saved workspace. It SHALL write a new results file and leave the original unchanged. A run whose workspace is missing SHALL be kept with its old outcome and listed.

#### Scenario: Rescoring the screen
- **WHEN** a user runs `rescore results/old/screen-1121/presets-llama.jsonl --runs-dir workspace/preset-runs/llama`
- **THEN** a new results file holds the same runs with fact checks added, and the original file is unchanged
