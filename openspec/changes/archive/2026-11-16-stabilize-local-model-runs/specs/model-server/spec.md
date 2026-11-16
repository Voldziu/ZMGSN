# Spec Delta

## ADDED Requirements

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

The command SHALL stop the active server even when interrupted. It SHALL write one results file per preset, and it SHALL resume finished runs when it is run again with the same configuration.

#### Scenario: Several presets in a row
- **WHEN** a user runs `make eval-presets PRESETS="llama gemma" REPEATS=3` with no server running
- **THEN** `llama` and then `gemma` are each served, evaluated three times and stopped, results go to `results/presets-llama.jsonl` and `results/presets-gemma.jsonl`, and a per-level summary is printed for each

#### Scenario: Interrupted run leaves no server behind
- **WHEN** the user presses Ctrl-C while a preset is being evaluated
- **THEN** the active server is stopped before the command exits

#### Scenario: Server already running
- **WHEN** a server already answers on port 8000
- **THEN** the command refuses to start and tells the user to stop it first

#### Scenario: Unknown preset
- **WHEN** `PRESETS` contains a name that is not a preset
- **THEN** the command exits before starting any server and lists the valid presets
