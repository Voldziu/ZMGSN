# Spec Delta

## MODIFIED Requirements

### Requirement: Completion token limit
The client SHALL send a maximum number of completion tokens with every request. It SHALL take the value from `LLM_MAX_TOKENS`, default 8192, and SHALL reject values below 1.

#### Scenario: Default token limit
- **WHEN** `LLM_MAX_TOKENS` is not set
- **THEN** each request asks for at most 8192 completion tokens

#### Scenario: Invalid token limit
- **WHEN** `LLM_MAX_TOKENS=0` is set
- **THEN** loading settings fails with a validation error

### Requirement: Results record the settings that change them
Each run record SHALL carry a configuration hash that covers:
- the step limit;
- the effective parallel-tool-call policy;
- the version of the tool definitions;
- the version of the outcome checks;
- the model and prompt settings.

Runs whose hashes differ SHALL NOT be treated as comparable.

#### Scenario: Step limit changes the hash
- **WHEN** two runs differ only in `AGENT_MAX_STEPS`
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
