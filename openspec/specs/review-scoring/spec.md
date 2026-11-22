# review-scoring Specification

## Purpose
Defines what an evaluation run must achieve to count as successful. That is a review that meets the task's format and is built from papers the agent actually read, so that scores reflect the work done and not only the shape of the file.

## Requirements

### Requirement: The task requires summaries from full text
The evaluation task SHALL require that each summary reflects the cited paper's full text, and SHALL say that a summary based only on search results or the abstract does not count. It SHALL state this as a property of the review, without prescribing which tools to use or in what order.

#### Scenario: Task text states the requirement
- **WHEN** the evaluation task prompt is rendered for any scenario
- **THEN** it states that each summary must reflect the paper's full text, not only its abstract, and it names no tool operation

### Requirement: Cited papers must have been read
Every evaluation run SHALL be scored with a `papers_read` criterion. The criterion SHALL pass only if, for every arXiv paper cited in the review, the run's own workspace contains that paper's extracted full text. It SHALL fail when any cited paper was not read in that run, and its detail SHALL name the unread papers.

#### Scenario: Review written from search results only
- **GIVEN** the agent searched arXiv and wrote a five-section review without downloading any paper
- **WHEN** the run is scored
- **THEN** `papers_read` fails, naming all five papers, and the run is not successful

#### Scenario: One paper could not be read
- **GIVEN** the agent read four papers and cited a fifth whose PDF failed to download
- **WHEN** the run is scored
- **THEN** `papers_read` fails and names the fifth paper

#### Scenario: All cited papers read
- **WHEN** every cited paper's full text was extracted in the run's workspace
- **THEN** `papers_read` passes

#### Scenario: Files from other runs do not count
- **WHEN** a cited paper's PDF exists only in the shared PDF cache, and its text was not extracted in this run
- **THEN** `papers_read` fails for that paper

### Requirement: Only completed runs succeed
A run SHALL count as a success only when its status is `completed` and every check passes. A run that hit the step limit, was stopped by the loop guard or failed SHALL NOT count as a success, even if every check passes, but its score SHALL still count as partial credit. The run record SHALL state whether the run succeeded, and the rule SHALL also apply to records written before it existed.

#### Scenario: Completed run passing every check
- **WHEN** a run ends with status `completed` and every check passes
- **THEN** it counts as a success with score 1.0

#### Scenario: Step limit after a complete review
- **GIVEN** the agent wrote a review that passes every check and then kept calling tools until the step limit
- **WHEN** the run is scored
- **THEN** it does not count as a success, and its score is 1.0

#### Scenario: Loop after a complete review
- **GIVEN** the agent wrote a review that passes every check and then requested the same review call until the loop guard stopped it
- **WHEN** the run is scored
- **THEN** its status is `loop`, it does not count as a success, and its score is 1.0

#### Scenario: Failed run keeps partial credit
- **WHEN** a run fails because the context window overflowed, and 6 of its 10 checks pass
- **THEN** it does not count as a success, and its score is 0.6

#### Scenario: Older results
- **GIVEN** a results file written before this rule, holding a `step_limit` run whose checks all passed
- **WHEN** the file is reported
- **THEN** that run does not count as a success

### Requirement: Scoring is retried on external failures
When scoring a finished run fails because an external service (arXiv) is unavailable, only the scoring SHALL be retried, with increasing waits, up to a fixed number of attempts. The agent run SHALL NOT be repeated. The run SHALL be recorded with status `error` only if every attempt fails. At the end of an evaluation, the command SHALL list the runs of the plan that have no scored record.

#### Scenario: Outage during scoring
- **GIVEN** the agent finished and arXiv answers 429 to the first two scoring attempts
- **WHEN** the third attempt succeeds
- **THEN** the run is recorded with its real status and outcome, and the agent ran only once

#### Scenario: Incomplete cells are reported
- **WHEN** one run of the plan still ends as `error` after all attempts
- **THEN** the evaluation's closing summary names that run as unscored

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
