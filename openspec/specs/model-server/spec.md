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

### Requirement: NVIDIA host prerequisites
The NVIDIA path SHALL require:
- an NVIDIA driver ≥ 580;
- the NVIDIA Container Toolkit, registered as a Docker runtime;
- a user who can run `docker` without `sudo`.

The documentation SHALL list these prerequisites and a check command for them.

#### Scenario: GPU visible in containers
- **WHEN** the prerequisites are met and a user runs `docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi`
- **THEN** the command lists the host GPU

#### Scenario: Missing toolkit
- **WHEN** the NVIDIA Container Toolkit is not configured for Docker
- **THEN** `make docker-up` fails because Docker cannot select a GPU device driver, and the documentation names the toolkit as the fix

### Requirement: Every CUDA preset starts within its budget
On a 16 GB card, each CUDA preset SHALL start, serve its model and answer a tool-calling request within its GPU-memory budget:

| Preset | Budget |
| --- | --- |
| `qwen`, `gemma`, `llama` | 12 GB |
| `qwen-mid` | 8 GB |
| `qwen-weak` | 4 GB |

The server's own tool check SHALL work with the `make` shipped on Linux as well as on macOS.

#### Scenario: Small budget starts
- **WHEN** a user runs `make docker-up PRESET=qwen-weak` on an RTX 5080
- **THEN** the server becomes ready using about 4 GB of GPU memory

#### Scenario: Tool check on Linux
- **WHEN** a user runs `make tool-check PRESET=qwen` with GNU Make 4.x while `qwen` is served
- **THEN** the request is valid JSON and the response contains a `get_weather` tool call

### Requirement: Multi-preset evaluation command
The project SHALL provide one command that, for each preset in a given list:
1. starts the preset's server on the current platform;
2. waits until the server is ready;
3. runs the evaluation with a given number of repeats;
4. stops the server.

The command SHALL stop the active server even when interrupted. It SHALL write one results file per preset, and it SHALL resume finished runs when it is run again with the same configuration. It SHALL keep each preset's run workspaces in a directory of that preset's own, so that one preset's runs never delete another preset's.

#### Scenario: Several presets in a row
- **WHEN** a user runs `make eval-presets PRESETS="llama gemma" REPEATS=3` with no server running
- **THEN** `llama` and then `gemma` are each served, evaluated three times and stopped, results go to `results/presets-llama.jsonl` and `results/presets-gemma.jsonl`, and a per-level summary is printed for each

#### Scenario: Workspaces kept per preset
- **WHEN** `llama` and then `gemma` have been evaluated on the same scenarios
- **THEN** `llama`'s reviews are still in `workspace/preset-runs/llama/` next to `gemma`'s in `workspace/preset-runs/gemma/`

#### Scenario: Interrupted run leaves no server behind
- **WHEN** the user presses Ctrl-C while a preset is being evaluated
- **THEN** the active server is stopped before the command exits

#### Scenario: Server already running
- **WHEN** a server already answers on port 8000
- **THEN** the command refuses to start and tells the user to stop it first

#### Scenario: Unknown preset
- **WHEN** `PRESETS` contains a name that is not a preset
- **THEN** the command exits before starting any server and lists the valid presets
