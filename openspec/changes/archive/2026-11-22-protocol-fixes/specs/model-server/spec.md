# Spec Delta

## ADDED Requirements

### Requirement: Each model is warmed up before its first run
Once a preset's server is ready, the multi-preset evaluation command SHALL send it one short tool-calling request before the first evaluation run. The warm-up SHALL NOT be recorded as a run and SHALL NOT enter any results file.

#### Scenario: Warm-up before gemma's first run
- **WHEN** `make eval-presets PRESETS=gemma` starts the `gemma` server
- **THEN** one warm-up request is answered before the first run starts, and the results file holds only evaluation runs

### Requirement: Server logs are kept per invocation
The multi-preset evaluation command SHALL write each preset's server log to a new file named after the preset and the start time. It SHALL NOT overwrite the log of an earlier invocation.

#### Scenario: Two invocations for the same preset
- **WHEN** `make eval-presets PRESETS=qwen` is run twice
- **THEN** `logs/` holds two server logs for `qwen`, one per invocation

## MODIFIED Requirements

### Requirement: Every CUDA preset starts within its budget
On a 16 GB card, each CUDA preset SHALL start, serve its model and answer a tool-calling request within its GPU-memory budget. `qwen-mid` SHALL be served in BF16 without online quantisation, so that the scale series does not gain precision as models get smaller.

| Preset | Budget |
| --- | --- |
| `qwen`, `gemma`, `llama`, `qwen-mid` | 12 GB |
| `qwen-weak` | 4 GB |

The server's own tool check SHALL work with the `make` shipped on Linux as well as on macOS.

#### Scenario: Small budget starts
- **WHEN** a user runs `make docker-up PRESET=qwen-weak` on an RTX 5080
- **THEN** the server becomes ready using about 4 GB of GPU memory

#### Scenario: Qwen 4B in BF16
- **WHEN** a user runs `make docker-up PRESET=qwen-mid` on an RTX 5080
- **THEN** the server loads `Qwen/Qwen3.5-4B` without quantisation, becomes ready within 12 GB of GPU memory, and answers the tool check

#### Scenario: Tool check on Linux
- **WHEN** a user runs `make tool-check PRESET=qwen` with GNU Make 4.x while `qwen` is served
- **THEN** the request is valid JSON and the response contains a `get_weather` tool call
