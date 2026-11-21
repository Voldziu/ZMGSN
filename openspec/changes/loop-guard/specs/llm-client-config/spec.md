# Spec Delta

## MODIFIED Requirements

### Requirement: Results record the settings that change them
Each run record SHALL carry a configuration hash that covers:
- the step limit;
- the loop-guard limits (`AGENT_MAX_REPEATED_CALLS` and `AGENT_MAX_TOOL_ERROR_STREAK`);
- the effective parallel-tool-call policy;
- the version of the tool definitions;
- the version of the outcome checks;
- the model and prompt settings.

Runs whose hashes differ SHALL NOT be treated as comparable.

#### Scenario: Step limit changes the hash
- **WHEN** two runs differ only in `AGENT_MAX_STEPS`
- **THEN** their configuration hashes differ

#### Scenario: Loop-guard limits change the hash
- **WHEN** two runs differ only in `AGENT_MAX_REPEATED_CALLS`, or only in `AGENT_MAX_TOOL_ERROR_STREAK`
- **THEN** their configuration hashes differ

#### Scenario: Parallel-call policy changes the hash
- **WHEN** two runs of `qwen` differ only in `LLM_PARALLEL_TOOL_CALLS`
- **THEN** their configuration hashes differ

#### Scenario: Tool definitions change the hash
- **WHEN** the tools of any level change (for example, level 2 replacing `append_to_review` with `write_review`)
- **THEN** runs made before and after the change have different configuration hashes

#### Scenario: Checks change the hash
- **WHEN** the outcome checks change (for example, adding `papers_read`)
- **THEN** runs scored before and after the change have different configuration hashes

## ADDED Requirements

### Requirement: Run records carry the loop-guard limits
Each run record SHALL store the two loop-guard limits it ran with. Records written before the loop guard existed SHALL still load, with both limits unknown.

#### Scenario: New record
- **WHEN** a run executes with default settings
- **THEN** its record stores 5 as the repeated-call limit and 10 as the error-streak limit

#### Scenario: Older record
- **WHEN** `report` reads a results file written before this change
- **THEN** its records load, and their loop-guard limits are unknown
