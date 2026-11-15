# model-server Specification

## Purpose
Provides a local OpenAI-compatible model server. It runs one model at a time, chosen from a fixed set of named presets, on NVIDIA GPUs (Docker) and on Apple Silicon (native). The evaluation client therefore works the same way on both platforms.

## Requirements

### Requirement: Fixed preset catalogue
The server SHALL offer exactly the presets `qwen`, `gemma`, `llama`, `qwen-mid` and `qwen-weak`, each with an NVIDIA variant and a macOS variant. Selecting any other preset name SHALL fail before a server is started.

#### Scenario: Both platforms define the same presets
- **WHEN** the NVIDIA and macOS preset sets are compared
- **THEN** both contain exactly `qwen`, `gemma`, `llama`, `qwen-mid` and `qwen-weak`

#### Scenario: Unknown preset is rejected
- **WHEN** a user runs a server command with `PRESET=mistral`
- **THEN** the command fails with a message that lists the valid presets

### Requirement: Platform-independent served identity
For a given preset, both platform variants SHALL use the preset name as the served model name. They SHALL also use the same tool-call parser, the same reasoning parser (where the preset has one) and the same chat template.

#### Scenario: Served name equals preset name
- **WHEN** preset `gemma` is running on either platform
- **THEN** `GET /v1/models` lists a model with id `gemma`

#### Scenario: Parsers match across platforms
- **WHEN** the NVIDIA and macOS variants of preset `llama` are compared
- **THEN** both use tool-call parser `llama3_json` and the same Llama 3.1 JSON chat template

### Requirement: OpenAI-compatible endpoint on port 8000
A running preset SHALL serve the OpenAI-compatible API at `http://localhost:8000/v1` on both platforms. It SHALL accept up to 32768 tokens of context.

#### Scenario: Chat completion endpoint reachable
- **WHEN** any preset is running
- **THEN** `POST http://localhost:8000/v1/chat/completions` with the preset name as `model` returns a completion

### Requirement: Structured tool calls
When a request supplies tools, every preset SHALL return the tool calls the model makes as structured `tool_calls` in the assistant message, not as raw text in `content`.

#### Scenario: Weather tool is called
- **GIVEN** a running preset
- **WHEN** the user asks "What is the weather in Paris?" and a `get_weather(city)` tool is supplied
- **THEN** the response message contains a `tool_calls` entry for `get_weather` with a `city` argument

### Requirement: Thinking mode controlled per request
For presets whose template supports reasoning, the server SHALL honour `chat_template_kwargs.enable_thinking` on each request. Presets whose template has no thinking mode SHALL ignore the flag and still answer.

#### Scenario: Thinking disabled
- **WHEN** a request to preset `qwen` sets `chat_template_kwargs.enable_thinking` to `false`
- **THEN** the response contains no reasoning content

#### Scenario: Flag ignored by Llama
- **WHEN** a request to preset `llama` sets `chat_template_kwargs.enable_thinking`
- **THEN** the request succeeds

### Requirement: Text-only serving on NVIDIA
NVIDIA presets SHALL accept no image or audio input, so that GPU memory is reserved for the language model.

#### Scenario: Multimodal input disabled
- **WHEN** an NVIDIA preset of a multimodal model (`qwen`, `gemma`, `qwen-mid`, `qwen-weak`) is running
- **THEN** the server is configured to accept no image or audio input

### Requirement: NVIDIA presets run in Docker within a 12 GB budget
On Linux, or on Windows with the WSL 2 backend, a preset SHALL be started with Docker Compose. It SHALL use the host NVIDIA GPU, and on a 16 GB card it SHALL stay within at most 12 GB of GPU memory.

#### Scenario: Start a preset with Compose
- **WHEN** a user runs `make docker-up PRESET=qwen` (or `docker compose --profile qwen up -d --build`)
- **THEN** the `qwen` container is built and started with GPU access and becomes `healthy` once the model has loaded

#### Scenario: Only one preset at a time
- **WHEN** a second preset is started while another is still running
- **THEN** it cannot bind port 8000 until the first preset is stopped with `make docker-down`

### Requirement: Shared weight cache
Downloaded model weights SHALL be stored in the host Hugging Face cache (`~/.cache/huggingface` unless `HF_CACHE_DIR` overrides it), so restarts and other presets reuse them. A `HF_TOKEN`, if set, SHALL be passed to the server for gated models.

#### Scenario: Weights survive a restart
- **WHEN** a preset is stopped and started again
- **THEN** the weights are not downloaded again

### Requirement: macOS presets run natively on the Apple GPU
On Apple Silicon a preset SHALL be served natively through vllm-metal using MLX checkpoints, started from the repo root. Starting the Docker path on macOS SHALL be refused, with a pointer to the native command.

#### Scenario: Serve a preset on macOS
- **WHEN** a user runs `make mac-install` once and then `make mac-serve PRESET=qwen-weak` from the repo root
- **THEN** the server starts in the foreground on port 8000 and serves `qwen-weak`

#### Scenario: Install falls back without Xcode
- **WHEN** the Homebrew install of vllm-metal fails
- **THEN** `make mac-install` installs vllm-metal with the upstream installer into `~/.venv-vllm-metal`, and `make mac-serve` uses that binary

#### Scenario: Docker refused on macOS
- **WHEN** a user runs `make docker-up` on macOS
- **THEN** the command exits with an error that tells the user to run `make mac-serve PRESET=<preset>`

### Requirement: Server check commands
The project SHALL provide platform-independent checks against a running server. They SHALL list served models, send one tool-calling request, and run one short evaluation that writes to its own results file.

#### Scenario: Smoke evaluation
- **WHEN** a user runs `make eval-smoke PRESET=qwen-weak` while `qwen-weak` is being served
- **THEN** one evaluation run completes against the server and writes `results/smoke-qwen-weak.jsonl` without touching other results
