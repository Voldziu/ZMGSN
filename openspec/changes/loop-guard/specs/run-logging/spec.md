# Spec Delta

## MODIFIED Requirements

### Requirement: Logging never changes runs or results
Logging SHALL NOT change the agent's behaviour, the run records or the configuration hash. A failure while writing a log line SHALL NOT stop a run.

#### Scenario: Results written before this change
- **WHEN** a results file written before this change is resumed
- **THEN** its finished runs are recognised and skipped, because the configuration hash is unchanged

#### Scenario: A warning does not stop the run
- **WHEN** a loop warning is written
- **THEN** the run continues until the agent finishes, the loop guard stops it or the step limit ends it

#### Scenario: Logging fails
- **WHEN** writing a step line raises an error
- **THEN** the run continues and its record is written as usual

## ADDED Requirements

### Requirement: A run stopped by the loop guard is logged
When the loop guard stops a run, the run's end line SHALL show the status `loop`, and the log SHALL get one WARNING line that names the signal and the call or tool that triggered it.

#### Scenario: Repeated review call
- **WHEN** the guard stops a run before the eighth identical `conduct_literature_review` call
- **THEN** the log has a warning naming that call and an end line with status `loop` and no success
