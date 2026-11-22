# Spec Delta

## ADDED Requirements

### Requirement: Explicit, identical sampling for every model
Every request SHALL carry an explicit `temperature`, `top_p` and `top_k` taken from the settings (`AGENT_TEMPERATURE`, `LLM_TOP_P`, `LLM_TOP_K`). The values SHALL be the same for every preset, so that no model samples with defaults from its own generation config. The run record SHALL carry all three values.

#### Scenario: Gemma does not use its own top-k
- **WHEN** a run of `gemma` is made with the default settings
- **THEN** its requests carry the configured `top_p` and `top_k`, not the `top_k: 64` from Gemma's generation config

#### Scenario: Qwen gets truncated sampling
- **WHEN** a run of `qwen` is made with the default settings
- **THEN** its requests carry the same `top_p` and `top_k` as every other preset

### Requirement: One tool call per reply for every preset
The client SHALL ask the server for at most one tool call per reply (`parallel_tool_calls: false`) for every preset, so that all models act under the same policy. `LLM_PARALLEL_TOOL_CALLS` SHALL override this policy.

#### Scenario: Llama gets one tool call per reply
- **WHEN** `LLM_MODEL=llama` and `LLM_PARALLEL_TOOL_CALLS` is not set
- **THEN** every request carries `parallel_tool_calls: false`

#### Scenario: Other presets get one tool call per reply too
- **WHEN** `LLM_MODEL=qwen` and `LLM_PARALLEL_TOOL_CALLS` is not set
- **THEN** every request carries `parallel_tool_calls: false`

#### Scenario: Explicit override
- **WHEN** `LLM_PARALLEL_TOOL_CALLS=true` is set with `LLM_MODEL=qwen`
- **THEN** every request carries `parallel_tool_calls: true`

## MODIFIED Requirements

### Requirement: Results record the settings that change them
Each run record SHALL carry a configuration hash that covers:
- the step limit;
- the loop-guard limits (`AGENT_MAX_REPEATED_CALLS` and `AGENT_MAX_TOOL_ERROR_STREAK`);
- the effective parallel-tool-call policy;
- the sampling settings (`temperature`, `top_p`, `top_k`);
- the completion token limit (`LLM_MAX_TOKENS`) and the thinking-mode flag;
- the served model's identity: checkpoint, quantisation, tool-call parser, a hash of the chat template, and the vLLM version;
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

#### Scenario: Sampling and token limit change the hash
- **WHEN** two runs differ only in `LLM_TOP_P`, only in `LLM_TOP_K`, or only in `LLM_MAX_TOKENS`
- **THEN** their configuration hashes differ

#### Scenario: Served model changes the hash
- **WHEN** `qwen-mid` is served once in FP8 and once in BF16, with all other settings equal
- **THEN** runs against the two servers have different configuration hashes

#### Scenario: Tool definitions change the hash
- **WHEN** the tools of any level change (for example, level 2 replacing `append_to_review` with `write_review`)
- **THEN** runs made before and after the change have different configuration hashes

#### Scenario: Checks change the hash
- **WHEN** the outcome checks change (for example, adding `papers_read`)
- **THEN** runs scored before and after the change have different configuration hashes

## REMOVED Requirements

### Requirement: Parallel tool calls follow a per-preset policy
**Reason**: Different policies per model made agent-call counts, tokens and time incomparable between models. The new requirement "One tool call per reply for every preset" replaces it.
**Migration**: None. `LLM_PARALLEL_TOOL_CALLS` still overrides the policy.
