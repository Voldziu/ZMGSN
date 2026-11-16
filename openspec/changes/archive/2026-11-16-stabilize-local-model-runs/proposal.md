# Proposal

## Why

The first real runs on both target machines (Apple M5 Pro and RTX 5080) showed that the vLLM migration was not yet usable for evaluation:

- On the RTX 5080, no CUDA preset started.
- `llama` failed every evaluation run.
- With 10 steps, the step limit measured how many tool calls a model makes per turn, not whether it can do the task. `gemma` scored 0.0 with 10 steps and 1.0 with 30.

## What Changes

- **CUDA server image.** It uses the CUDA 13.0 `devel` base again, with `nvcc`/`crt`/`nvvm` pinned to 13.0. FlashInfer compiles and links its kernels when the server starts, and needs a compiler and toolkit that match torch's CUDA. Already applied: `92c4de3`, `fd1ce13`.
- **CUDA presets.**
  - All presets cap `max-num-seqs` at 8. The evaluation sends one request at a time, and vLLM's default of 256 does not fit Qwen3.5's cache in small budgets. Already applied: `20f3aaf`.
  - `qwen-mid` gets an 8 GB budget (`0.5`) instead of 6 GB. Already applied: `250d181`.
- **`make tool-check`** builds its JSON body in a way that works with GNU Make 4.x as well as macOS make 3.81. Already applied: `60b2f1d`.
- **Host prerequisite.** NVIDIA Container Toolkit, configured for Docker, is documented as a requirement.
- **Parallel tool calls.** The client asks the server for at most one tool call per reply for presets whose chat template cannot render several (`llama`), and a setting can override this. The other presets keep parallel tool calls.
- **Tool arguments.** A tool argument that the schema declares as a list or object but that arrives as a JSON-encoded string is decoded before validation, the same way for every model.
- **BREAKING** The default agent step limit rises from 10 to 30. Runs made with 10 steps are not comparable, and their config hash differs.

## Capabilities

### New Capabilities

- `agent-tool-calling`: how the agent executes tool calls. Covers the default step limit and tolerance for JSON-encoded arguments.

### Modified Capabilities

- `llm-client-config`: adds the parallel-tool-call policy per preset.
- `model-server`: adds the NVIDIA host prerequisites and requires every CUDA preset to start within its documented budget.

## Impact

- **Code:** `src/tool_eval/config.py`, `src/tool_eval/llm.py`, `src/tool_eval/agent.py`, `src/tool_eval/evaluation/runner.py` (config hash).
- **Server assets** (already applied): `src/tool_eval/server/Dockerfile`, `src/tool_eval/server/presets/cuda/*.yaml`, `pyproject.toml`, `uv.lock`, `Makefile`.
- **Docs:** README, `.env.example`, `docs/superpowers/specs/2026-11-09-vllm-server-design.md`.
- **Results:** every run made after this change has a new config hash, because both the step limit and the hash payload change.
