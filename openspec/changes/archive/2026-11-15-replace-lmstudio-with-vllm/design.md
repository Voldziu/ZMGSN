# Design

## Context

See proposal.md for the reasons behind this change. The client already used `ChatOpenAI` against an OpenAI-compatible endpoint, so only the endpoint and the per-request options change. The two target machines are:

- an RTX 5080 box (16 GB VRAM, of which at most 12 GB is used), running Linux or Windows with WSL 2;
- an Apple Silicon Mac. The original spec assumed 32 GB of unified memory; the development Mac has 24 GB.

The original long-form design and its implementation plan are kept in `docs/superpowers/specs/2026-11-09-vllm-server-design.md` and `docs/superpowers/plans/2026-11-09-vllm-server.md`.

## Goals / Non-Goals

**Goals:**
- Run one model at a time, picked by preset name, with the same served name and parsers on both platforms.
- Keep the Python client platform-agnostic: the only thing that differs between platforms is the `LLM_MODEL` value.
- Keep `uv sync` / `uv lock` working on macOS even though vLLM CUDA wheels exist only for Linux.

**Non-Goals:**
- Serving several presets at once (they all use port 8000).
- Comparing CUDA and Metal results directly. The 4-bit formats differ, so Mac runs are for development only.
- Running vLLM in Docker on macOS (CPU only, no Metal).

## Decisions

### One `vllm serve --config` YAML per preset and platform
`src/tool_eval/server/presets/{cuda,metal}/<name>.yaml`. The file stem is the `served-model-name`. Keeping the configs in files means Compose, the Makefile and the README all point at one source of truth.
*Alternative:* CLI flags in Compose `command`. Rejected because the flags would be duplicated in the macOS commands.

### Model catalogue

| Preset | CUDA checkpoint | CUDA budget | Metal checkpoint | Tool / reasoning parser | Template |
| --- | --- | --- | --- | --- | --- |
| `qwen` | `cyankiwi/Qwen3.5-9B-AWQ-4bit` | 0.75 | `mlx-community/Qwen3.5-9B-MLX-4bit` | `qwen3_coder` / `qwen3` | model default |
| `gemma` | `google/gemma-4-12B-it-qat-w4a16-ct` | 0.75 | `mlx-community/gemma-4-12B-it-4bit` | `gemma4` / `gemma4` | `tool_chat_template_gemma4.jinja` |
| `llama` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | 0.75 | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | `llama3_json` / — | `tool_chat_template_llama3.1_json.jinja` |
| `qwen-mid` | `Qwen/Qwen3.5-4B` + `quantization: fp8` | 0.375 | `mlx-community/Qwen3.5-4B-MLX-4bit` | `qwen3_coder` / `qwen3` | model default |
| `qwen-weak` | `Qwen/Qwen3.5-0.8B` (BF16) | 0.25 | `mlx-community/Qwen3.5-0.8B-MLX-bf16` | `qwen3_coder` / `qwen3` | model default |

- Every preset uses a context of 32768 tokens.
- CUDA presets turn off multimodal input: `language-model-only` for Qwen, and `limit-mm-per-prompt` image/audio 0 for Gemma.
- Llama 3.1-8B is the newest Llama that fits. Qwen3.6/3.8 start at 27B, so all Qwen presets use Qwen3.5.
- As of 2026-11-15, all ten repositories were public and not gated. Weight sizes ranged from 1.7 to 10.3 GB.

### NVIDIA: CUDA 13 base image with the toolkit from pip
- `nvidia/cuda:13.0.2-base-ubuntu24.04` plus uv 0.13.7 and Python 3.12.
- vLLM comes from the `llm` group with `uv sync --frozen --only-group llm`.
- `nvcc` and the CUDA headers come from torch's `nvidia/cu13` wheels (`CUDA_HOME` points there), and `g++` comes from apt.
- This cut the image from 16.1 GB to 10.2 GB compared with the `devel` base.
- `vllm download-kernels` was removed from the build because it needs a GPU.

*Alternative:* the official `vllm/vllm-openai` image. Rejected so that the vLLM version stays pinned in `uv.lock`, together with the project's own templates and presets.

### NVIDIA: one Compose service per preset, each in its own profile
`docker compose --profile <preset> up -d --build`. A shared `x-vllm-base` anchor holds:
- GPU reservation and `ipc: host`;
- the bind mount `${HF_CACHE_DIR:-~/.cache/huggingface}` and `HF_TOKEN`;
- the `/health` healthcheck, with `start_period: 30m` to cover the first weight download.

### macOS: native vllm-metal
- `brew tap vllm-project/vllm-metal … && brew install …`.
- When brew fails (for example because there is no current Xcode), the Makefile falls back to the `install.sh` venv at `~/.venv-vllm-metal`.
- `make mac-serve` picks whichever `vllm` it finds first.
- `GLOO_SOCKET_IFNAME=lo0` avoids a startup stall when the hostname does not resolve.
- Metal presets reference templates by repo-relative path, so the server must be started from the repo root.

### `llm` dependency group, Linux only, conflicting with `dev`
`vllm==0.33.0; sys_platform == 'linux' and python_version < '3.15'`, with `[tool.uv] conflicts = [[dev, llm]]` because vLLM's pins clash with `langgraph-cli`. macOS never installs it.

### Client sends `max_completion_tokens` and `chat_template_kwargs.enable_thinking` on every request
- `max_tokens` is 4096, following MCP-GRANITE.
- Thinking is off by default. The flag is read by the Qwen3.5 and Gemma 4 templates and ignored by Llama's template.
- The evaluation CLI keeps its own sampling settings (`--temperature 0.7`, 10 repeats).

## Risks / Trade-offs

- [Mac has 24 GB, not the 32 GB the spec assumed.] → The largest Metal weights are 6.7 GB (`gemma`). Check memory headroom when serving `gemma` and `qwen` at 32k context.
- [No current Xcode on the dev Mac, so `brew install vllm-metal` may fail.] → `make mac-install` falls back to `install.sh`, which needs no compiler.
- [Metal Qwen presets do not set `language-model-only`.] vllm-metal serves Qwen3.5 MLX checkpoints with native multimodal support, so the vision tower may load and use memory. → Add `language-model-only` to the Metal Qwen presets if it is supported and memory turns out to be tight.
- [`mlx-community/gemma-4-12B-it-4bit` declares a vision/audio config but ships no vision weights.] → vllm-metal documents that it then serves text only, which is what the preset wants.
- [`qwen-mid` FP8 may not fit in 6 GB.] → Fall back to `sanskar003/Qwen3.5-4B-AWQ`. That repo is gated (`auto`), so it needs `HF_TOKEN`.
- [The Llama 3.1 tool template raises an error on more than one tool call per assistant message.] → Check during manual verification whether the agent emits parallel calls with `llama`.
- [GPU-utilisation fractions assume a 16 GB card.] → They must be recomputed for any other GPU.

## Migration Plan

1. Update `.env`: set `LLM_BASE_URL=http://localhost:8000/v1`, `LLM_API_KEY=EMPTY` and `LLM_MODEL=<preset>`. Optionally set `LLM_MAX_TOKENS`, `LLM_ENABLE_THINKING`, `HF_TOKEN` and `HF_CACHE_DIR`.
2. Start a preset: `make mac-install && make mac-serve PRESET=<p>` on macOS, or `make docker-up PRESET=<p>` on NVIDIA.
3. Check the server with `make models`, `make tool-check` and `make eval-smoke`.

Rollback: revert the client defaults and point `LLM_BASE_URL` back at LM Studio. The server assets are additive.

## Open Questions

- Does every Metal preset start and return structured `tool_calls` on the 24 GB Mac? This is still unverified, and it is tracked in tasks.md.
