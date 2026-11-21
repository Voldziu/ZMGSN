# Spec Delta

## MODIFIED Requirements

### Requirement: Each evaluation run is an MLflow run
With tracing enabled, each evaluation run SHALL be logged as one MLflow run in the configured experiment (default `tool-eval`). The MLflow run SHALL carry:
- **parameters:** model, level, scenario, repeat, seed, order seed, position in the block, step limit, loop-guard limits, temperature, prompt version, tool-set version, checks version and config hash;
- **metrics:** score, success (under the success rule), model calls, tool calls, tool errors, malformed tool calls, tokens (including the total and the first agent call's input tokens), model and tool time, and wall time;
- **tags:** status and error;
- **artifacts:** the review file and the run record.

A measurement that the run's record does not know SHALL NOT be logged.

#### Scenario: Compare models in the UI
- **WHEN** `qwen` and `llama` were evaluated with tracing enabled
- **THEN** their runs appear in one experiment with `model` and `level` parameters and a `score` metric, so that they can be filtered and compared

#### Scenario: Review attached
- **WHEN** a run wrote `reviews/review.md`
- **THEN** the file is stored as an artifact of the run's MLflow run

#### Scenario: Step-limit run is not a success
- **WHEN** a run that hit the step limit passed every check
- **THEN** its `success` metric is 0 and its `score` metric is 1

#### Scenario: Prompt overhead and time
- **WHEN** a run is logged
- **THEN** its metrics include the total tokens, the first agent call's input tokens, the agent and tool-internal model time and the tool time

#### Scenario: Loop run
- **WHEN** the loop guard stopped a run
- **THEN** its `status` tag is `loop`, its `error` tag names the signal, and its `success` metric is 0
