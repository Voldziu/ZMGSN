# Spec Delta

## ADDED Requirements

### Requirement: Scoring is retried on external failures
When scoring a finished run fails because an external service (arXiv) is unavailable, only the scoring SHALL be retried, with increasing waits, up to a fixed number of attempts. The agent run SHALL NOT be repeated. The run SHALL be recorded with status `error` only if every attempt fails. At the end of an evaluation, the command SHALL list the runs of the plan that have no scored record.

#### Scenario: Outage during scoring
- **GIVEN** the agent finished and arXiv answers 429 to the first two scoring attempts
- **WHEN** the third attempt succeeds
- **THEN** the run is recorded with its real status and outcome, and the agent ran only once

#### Scenario: Incomplete cells are reported
- **WHEN** one run of the plan still ends as `error` after all attempts
- **THEN** the evaluation's closing summary names that run as unscored

## MODIFIED Requirements

### Requirement: The task requires summaries from full text
The evaluation task SHALL require that each summary reflects the cited paper's full text, and SHALL say that a summary based only on search results or the abstract does not count. It SHALL state this as a property of the review, without prescribing which tools to use or in what order.

#### Scenario: Task text states the requirement
- **WHEN** the evaluation task prompt is rendered for any scenario
- **THEN** it states that each summary must reflect the paper's full text, not only its abstract, and it names no tool operation
