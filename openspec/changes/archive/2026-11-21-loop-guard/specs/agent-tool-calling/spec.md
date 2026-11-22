# Spec Delta

## ADDED Requirements

### Requirement: Repeated identical calls stop the run
When the agent's model requests the same tool call, meaning the same tool with the same arguments, for the `AGENT_MAX_REPEATED_CALLS`-th time in a row (default 8), the run SHALL stop before that call executes and SHALL be recorded with status `loop`. Tool calls made inside a tool SHALL NOT count. A value of 0 SHALL switch this check off.

#### Scenario: Same review requested again and again
- **WHEN** a level-3 agent calls `conduct_literature_review` with the same query seven times in a row and then requests it an eighth time
- **THEN** the eighth call does not execute, and the run is recorded as `loop` with an error naming `conduct_literature_review` and its arguments

#### Scenario: Different arguments
- **WHEN** the agent calls `search_arxiv` ten times in a row with a different query each time
- **THEN** the run is not stopped by this check

#### Scenario: Another call in between
- **WHEN** the agent calls `read_file` with the same path four times, calls `search_arxiv` once, and then calls `read_file` with that path four more times
- **THEN** the run is not stopped by this check

#### Scenario: Several identical calls in one reply
- **GIVEN** a model that may request several tool calls in one reply
- **WHEN** one reply requests the same call eight times
- **THEN** the run stops before any of those calls executes and is recorded as `loop`

#### Scenario: Check switched off
- **WHEN** `AGENT_MAX_REPEATED_CALLS=0` and the agent repeats one call until the step limit
- **THEN** the run is recorded as `step_limit`

### Requirement: A streak of failed tool calls stops the run
When `AGENT_MAX_TOOL_ERROR_STREAK` tool calls of the agent in a row fail (default 10), the run SHALL stop before the next reply of the agent's model and SHALL be recorded with status `loop`. A tool call that succeeds SHALL reset the count. A value of 0 SHALL switch this check off.

#### Scenario: Invented queries that all fail
- **WHEN** a level-3 agent calls `conduct_literature_review` with a different query each time and ten calls in a row fail
- **THEN** the agent's model is not called again, and the run is recorded as `loop` with an error naming the streak and the last failed tool

#### Scenario: A success resets the count
- **WHEN** nine tool calls fail, the next one succeeds and nine more fail
- **THEN** the run is not stopped by this check

#### Scenario: Agent finishes after the streak
- **WHEN** the tenth failed call is the agent's last tool call and its model would not reply again
- **THEN** the run is recorded as `loop`, because the guard stops it before that reply

### Requirement: The step limit still applies
A run that the loop guard does not stop SHALL still end at the step limit with status `step_limit`. When both limits would apply, the first one reached SHALL decide the status.

#### Scenario: Slow loop with varying arguments
- **WHEN** the agent alternates between two calls that succeed until it has made 100 model calls
- **THEN** the run is recorded as `step_limit`
