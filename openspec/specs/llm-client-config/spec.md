# llm-client-config Specification

## Purpose
Defines how the evaluation client connects to the model server and which generation options it sends with every request. Settings come from the environment or `.env`.

## Requirements

### Requirement: Local vLLM connection defaults
With no overrides, the client SHALL connect to `http://localhost:8000/v1` with API key `EMPTY` and model `qwen`. `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` SHALL override these defaults.

#### Scenario: Defaults without .env
- **WHEN** settings are loaded with no `LLM_*` variables set
- **THEN** the base URL is `http://localhost:8000/v1`, the API key is `EMPTY` and the model is `qwen`

#### Scenario: Model selected by preset name
- **WHEN** `LLM_MODEL=gemma` is set
- **THEN** requests are sent with `model` set to `gemma`

### Requirement: Completion token limit
The client SHALL send a maximum number of completion tokens with every request. It SHALL take the value from `LLM_MAX_TOKENS`, default 4096, and SHALL reject values below 1.

#### Scenario: Default token limit
- **WHEN** `LLM_MAX_TOKENS` is not set
- **THEN** each request asks for at most 4096 completion tokens

#### Scenario: Invalid token limit
- **WHEN** `LLM_MAX_TOKENS=0` is set
- **THEN** loading settings fails with a validation error

### Requirement: Thinking-mode flag
The client SHALL send `chat_template_kwargs.enable_thinking` with every request, using the value of `LLM_ENABLE_THINKING` (default `false`).

#### Scenario: Thinking off by default
- **WHEN** `LLM_ENABLE_THINKING` is not set
- **THEN** each request carries `chat_template_kwargs: {"enable_thinking": false}`

#### Scenario: Thinking enabled
- **WHEN** `LLM_ENABLE_THINKING=true` is set
- **THEN** each request carries `chat_template_kwargs: {"enable_thinking": true}`

### Requirement: Unreachable server is reported per run
If the model server cannot be reached, the evaluation SHALL record the failure for the affected run and continue, rather than stop the whole evaluation.

#### Scenario: Server down during evaluation
- **WHEN** the model server is not running and an evaluation run starts
- **THEN** the run is recorded with a connection error and the evaluation goes on to the next run

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
