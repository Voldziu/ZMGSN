# Spec Delta

## RENAMED Requirements

- FROM: `### Requirement: Default step limit of 30`
- TO: `### Requirement: Default step limit`

## MODIFIED Requirements

### Requirement: Default step limit
A run SHALL allow at most 100 model calls by default. `AGENT_MAX_STEPS` SHALL override the limit. A run that reaches the limit without finishing SHALL be recorded with status `step_limit`.

#### Scenario: Default limit
- **WHEN** `AGENT_MAX_STEPS` is not set
- **THEN** a run may take up to 100 model calls before it is stopped

#### Scenario: One-tool-per-turn model finishes
- **GIVEN** a model that calls one tool per reply and needs about 33 calls for the level-1 scenario
- **WHEN** it runs with the default limit
- **THEN** the run can end with status `completed`

#### Scenario: Runaway run is stopped
- **WHEN** a model keeps calling tools without finishing
- **THEN** the run stops after 100 model calls and is recorded as `step_limit`

### Requirement: JSON-encoded list and object arguments are decoded
Before a tool runs, an argument whose declared type is a list or an object, but whose value is a string, SHALL be decoded if the string holds JSON or a Python literal of exactly that type, and the decoded value SHALL replace the string. Any other argument SHALL be passed through unchanged.

#### Scenario: Stringified list is accepted
- **WHEN** the model calls `format_apa_citation` with `authors` set to the string `["Ada Lovelace", "Alan Turing"]`
- **THEN** the tool receives a list of two names and formats the citation

#### Scenario: Python list literal is accepted
- **WHEN** the model calls `format_apa_citation` with `authors` set to the string `['E. Kirkinis', 'A. V. Andreev']`
- **THEN** the tool receives a list of two names and formats the citation

#### Scenario: Wrong type still fails
- **WHEN** the model passes `authors` as the string `Ada Lovelace` (neither a JSON list nor a Python list literal)
- **THEN** the tool call fails validation and is counted as a tool error

#### Scenario: Strings stay strings
- **WHEN** an argument declared as a string contains text that looks like JSON
- **THEN** the tool receives the original string

## ADDED Requirements

### Requirement: Truncated final replies fail the run
If the agent's final reply was cut off by the completion-token limit and contains no tool call, the run SHALL be recorded with status `failed`. Its error SHALL say that the reply hit the token limit. Such a run SHALL NOT be recorded as `completed`.

#### Scenario: Review cut off mid-tool-call
- **GIVEN** the model starts a `write_review` call whose arguments do not fit in the token limit
- **WHEN** the server stops the reply at the limit and the run ends without a parsed tool call
- **THEN** the run is recorded as `failed` with an error naming the token limit

#### Scenario: Normal final answer
- **WHEN** the final reply ends normally (`finish_reason` `stop`)
- **THEN** the run is recorded as `completed`
