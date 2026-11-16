# Spec Delta

## ADDED Requirements

### Requirement: Parallel tool calls follow a per-preset policy
The client SHALL ask the server for at most one tool call per reply when the active preset cannot render several tool calls in one message (`llama`). It SHALL leave parallel tool calls to the server's default for every other preset. `LLM_PARALLEL_TOOL_CALLS` SHALL override this policy for any preset.

#### Scenario: Llama gets one tool call per reply
- **WHEN** `LLM_MODEL=llama` and `LLM_PARALLEL_TOOL_CALLS` is not set
- **THEN** every request carries `parallel_tool_calls: false`

#### Scenario: Other presets keep parallel calls
- **WHEN** `LLM_MODEL=qwen` and `LLM_PARALLEL_TOOL_CALLS` is not set
- **THEN** requests do not restrict parallel tool calls

#### Scenario: Explicit override
- **WHEN** `LLM_PARALLEL_TOOL_CALLS=false` is set with `LLM_MODEL=qwen`
- **THEN** every request carries `parallel_tool_calls: false`

### Requirement: Results record the settings that change them
Each run record SHALL carry a configuration hash that covers the step limit and the effective parallel-tool-call policy, along with the model and prompt settings. Runs whose hashes differ SHALL NOT be treated as comparable.

#### Scenario: Step limit changes the hash
- **WHEN** two runs differ only in `AGENT_MAX_STEPS`
- **THEN** their configuration hashes differ

#### Scenario: Parallel-call policy changes the hash
- **WHEN** two runs of `qwen` differ only in `LLM_PARALLEL_TOOL_CALLS`
- **THEN** their configuration hashes differ
