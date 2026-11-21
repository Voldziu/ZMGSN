# Spec Delta

## MODIFIED Requirements

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
