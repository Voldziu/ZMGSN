# Spec Delta

## Purpose

Defines how the evaluation agent executes tool calls: how many model calls a run may take, and how tool arguments from the model are prepared before the tool runs.

## ADDED Requirements

### Requirement: Default step limit of 30
A run SHALL allow at most 30 model calls by default. `AGENT_MAX_STEPS` SHALL override the limit. A run that reaches the limit without finishing SHALL be recorded with status `step_limit`.

#### Scenario: Default limit
- **WHEN** `AGENT_MAX_STEPS` is not set
- **THEN** a run may take up to 30 model calls before it is stopped

#### Scenario: One-tool-per-turn model finishes
- **GIVEN** a model that calls one tool per reply and needs about 25 calls for the level-1 scenario
- **WHEN** it runs with the default limit
- **THEN** the run can end with status `completed`

#### Scenario: Runaway run is stopped
- **WHEN** a model keeps calling tools without finishing
- **THEN** the run stops after 30 model calls and is recorded as `step_limit`

### Requirement: JSON-encoded list and object arguments are decoded
Before a tool runs, an argument whose declared type is a list or an object, but whose value is a string holding JSON of that type, SHALL be replaced by the decoded value. Any other argument SHALL be passed through unchanged.

#### Scenario: Stringified list is accepted
- **WHEN** the model calls `format_apa_citation` with `authors` set to the string `["Ada Lovelace", "Alan Turing"]`
- **THEN** the tool receives a list of two names and formats the citation

#### Scenario: Wrong type still fails
- **WHEN** the model passes `authors` as the string `Ada Lovelace` (not a JSON list)
- **THEN** the tool call fails validation and is counted as a tool error

#### Scenario: Strings stay strings
- **WHEN** an argument declared as a string contains text that looks like JSON
- **THEN** the tool receives the original string
