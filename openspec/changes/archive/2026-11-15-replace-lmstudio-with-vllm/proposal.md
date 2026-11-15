# Proposal

## Why

The evaluation used LM Studio, a desktop GUI. In LM Studio the model was picked by hand, so runs could not be scripted or repeated on the NVIDIA thesis machine. A vLLM server makes the model set reproducible. It serves the same models through one OpenAI-compatible API on NVIDIA (Docker) and on Apple Silicon (native `vllm-metal`). It also gives explicit tool-call and reasoning parsers per model.

## What Changes

- **BREAKING** Replace LM Studio (`http://localhost:1234/v1`, key `lm-studio`, model `local-model`) with a vLLM server on `http://localhost:8000/v1` (key `EMPTY`). Existing `.env` files must be updated.
- Add five model presets, `qwen`, `gemma`, `llama`, `qwen-mid` and `qwen-weak`. Each preset has a CUDA variant and a Metal variant that share the served name, tool-call parser, reasoning parser and chat template.
- NVIDIA path: a CUDA 13 server image (vLLM 0.33.0) and one Docker Compose profile per preset, with a shared Hugging Face weight cache and a health check.
- macOS path: install `vllm-metal` natively (Homebrew, or `install.sh` as a fallback) and serve a Metal preset from the repo root. Docker on macOS is refused because it has no GPU access.
- Add a `Makefile` cheat sheet with these steps: install/serve on macOS; up/status/logs/down with Docker; then `models`, `tool-check` and `eval-smoke` checks against the running server.
- Client: new settings `LLM_MAX_TOKENS` (default 4096) and `LLM_ENABLE_THINKING` (default `false`). Every request sends both.
- Remove LM Studio wording from the README, `.env.example` and tests.

## Capabilities

### New Capabilities

- `model-server`: the local OpenAI-compatible model server. Covers the preset catalogue, how a preset is started on NVIDIA (Docker Compose) and on macOS (vllm-metal), and how a running server is checked.
- `llm-client-config`: how the evaluation client reaches the model server. Covers connection defaults, generation limits and the thinking-mode switch.

### Modified Capabilities

None. The project has no specs yet.

## Impact

- Code: `src/tool_eval/config.py`, `src/tool_eval/llm.py`.
- New server assets: `src/tool_eval/server/{Dockerfile,presets/{cuda,metal}/*.yaml,templates/*.jinja}`, `docker-compose.yml`, `.dockerignore`, `Makefile`.
- Dependencies: a new `llm` uv group (`vllm==0.33.0`, Linux only). It conflicts with `dev` and is installed only in the server image. `pyyaml` and `types-pyyaml` were added to `dev`.
- Host requirements: NVIDIA driver ≥ 580 for the Docker path; macOS 15+ on Apple Silicon for the Metal path.
- Docs: README "Model server" section, `.env.example`.
- Out of scope: automatic model switching in the evaluation runner, and vLLM in Docker on macOS.
