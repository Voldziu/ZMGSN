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
