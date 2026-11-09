# vLLM server instead of LM Studio

## Goal

Replace LM Studio with a vLLM server, one model at a time, chosen from five presets:

- **NVIDIA (Linux, Windows via Docker Desktop + WSL2):** Docker Compose on an RTX 5080
  (16 GB, max 12 GB used), 4-bit CUDA checkpoints.
- **macOS (Apple Silicon):** native `vllm-metal` installed with Homebrew, GPU via MLX, 4-bit MLX
  checkpoints. Docker on macOS has no GPU access, so the Mac path does not use Docker.

Both paths serve the same OpenAI-compatible API on port 8000 with the same served model names and
tool parsers, so the Python client is platform-agnostic.

## Decisions

| Topic | Decision |
| --- | --- |
| Preset location | One `vllm serve --config` YAML per preset and platform: `src/tool_eval/server/presets/{cuda,metal}/<name>.yaml` |
| Model switching (NVIDIA) | One Compose service per preset with its own profile (`docker compose --profile qwen up`), `command: [--config, /app/presets/cuda/<name>.yaml]` |
| Model switching (macOS) | `vllm serve --config src/tool_eval/server/presets/metal/<name>.yaml` from the repo root |
| Compose location | Repo root; build context `.`, Dockerfile in `src/tool_eval/server/` |
| Base image | `nvidia/cuda:13.0.2-base-ubuntu24.04` + uv; vLLM 0.33.0 installed from the `llm` dependency group (its PyPI torch 2.14.0 is built for CUDA 13.0; host driver ≥ 580) |
| macOS install | `brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal && brew install vllm-project/vllm-metal/vllm-metal` |
| Weights | Docker: bind mount `${HF_CACHE_DIR:-~/.cache/huggingface}` → `/root/.cache/huggingface`; macOS uses the same host cache natively; `HF_TOKEN` from `.env` |
| Model on the client | Explicit `LLM_MODEL` in `.env`, equal to the preset's `served-model-name` (same on both platforms) |
| Thinking mode | `LLM_ENABLE_THINKING` setting, default `false`, sent per request as `chat_template_kwargs` |
| Agent controls | `max_tokens` 4096 as in MCP-GRANITE (arXiv:2609.24161). Evaluation keeps its own sampling (CLI `--temperature` 0.7, 10 repeats) to measure variance; `AGENT_TEMPERATURE=0` and 10 max steps stay for single runs |

## Presets

Every preset file sets `model`, `served-model-name`, `max-model-len: 32768`,
`enable-auto-tool-choice: true`, `tool-call-parser` and, where applicable, `reasoning-parser` and
`chat-template`. Host and port come from the caller (Dockerfile entrypoint / README command).

Shared per preset:

| Preset / served name | Tool parser | Reasoning parser | Chat template |
| --- | --- | --- | --- |
| `qwen`, `qwen-mid`, `qwen-weak` | `qwen3_coder` | `qwen3` | model default |
| `gemma` | `gemma4` | `gemma4` | `tool_chat_template_gemma4.jinja` |
| `llama` | `llama3_json` | — | `tool_chat_template_llama3.1_json.jinja` |

CUDA (`presets/cuda/`), budget = `gpu-memory-utilization` × 16 GB, multimodal inputs disabled
(`language-model-only` for Qwen, `limit-mm-per-prompt` image/audio 0 for Gemma):

| Preset | HF repo | Quantization | Budget (util) |
| --- | --- | --- | --- |
| `qwen` | `cyankiwi/Qwen3.5-9B-AWQ-4bit` | AWQ 4-bit (compressed-tensors) | 12 GB (0.75) |
| `gemma` | `google/gemma-4-12B-it-qat-w4a16-ct` | QAT W4A16 | 12 GB (0.75) |
| `llama` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | AWQ 4-bit | 12 GB (0.75) |
| `qwen-mid` | `Qwen/Qwen3.5-4B` | FP8 online (`quantization: fp8`) | 6 GB (0.375) |
| `qwen-weak` | `Qwen/Qwen3.5-0.8B` | BF16 | 4 GB (0.25) |

Metal (`presets/metal/`), Mac with 32 GB unified memory, `vllm-metal` default memory fraction:

| Preset | HF repo | Quantization |
| --- | --- | --- |
| `qwen` | `mlx-community/Qwen3.5-9B-MLX-4bit` | MLX 4-bit |
| `gemma` | `mlx-community/gemma-4-12B-it-4bit` | MLX 4-bit (text-only checkpoint) |
| `llama` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | MLX 4-bit |
| `qwen-mid` | `mlx-community/Qwen3.5-4B-MLX-4bit` | MLX 4-bit |
| `qwen-weak` | `mlx-community/Qwen3.5-0.8B-MLX-bf16` | BF16 |

Notes:

- Llama 3.1-8B is the newest Llama that fits; Meta has released no smaller model since (Llama 4 starts above 100B total parameters).
- Qwen3.6/3.8 start at 27B, so all Qwen presets use Qwen3.5.
- If `qwen-mid` does not fit in 6 GB with FP8, fall back to `sanskar003/Qwen3.5-4B-AWQ`.
- Utilization fractions assume a 16 GB card; on another GPU they must be recomputed.
- CUDA and Metal presets use different 4-bit formats (AWQ/W4A16 vs MLX) of the same models. Mac
  results are for development; thesis comparisons use the RTX runs.
- In `vllm-metal`, Qwen3.5, Llama 3 and Gemma 4 (text path) are listed as supported.
- The Llama 3.1 tool template raises on more than one tool call per assistant message; manual
  verification checks whether the agent ever emits parallel calls with the `llama` preset.

## Components

### `pyproject.toml` — `llm` dependency group

- `vllm==0.33.0` with marker `sys_platform == 'linux' and python_version < '3.15'` so `uv lock`/`uv sync`
  keep working on macOS and the universal lock stays solvable (vLLM supports Python < 3.15).
- No custom torch index: vLLM 0.33.0 pins `torch==2.14.0`, whose PyPI Linux wheels bundle CUDA 13.0.
- Not part of default groups; `uv sync` on macOS installs nothing from it.

### `src/tool_eval/server/Dockerfile`

- `FROM nvidia/cuda:13.0.2-base-ubuntu24.04` (nvcc and CUDA headers come from torch's pip `nvidia/cu13`; `CUDA_HOME` points there, `g++` from apt).
- uv 0.13.7 copied from `ghcr.io/astral-sh/uv`, Python 3.12 installed by uv.
- `COPY pyproject.toml uv.lock` → `uv sync --frozen --only-group llm --no-install-project`, then
  `vllm download-kernels` (precompiled FlashInfer kernels, avoids minutes of JIT on Blackwell).
- `COPY src/tool_eval/server/templates /app/templates` and
  `COPY src/tool_eval/server/presets/cuda /app/presets/cuda`.
- `ENTRYPOINT ["vllm", "serve", "--host", "0.0.0.0", "--port", "8000"]`.

### `src/tool_eval/server/presets/`

`cuda/<name>.yaml` and `metal/<name>.yaml` for the five presets, per the tables above. CUDA presets
reference templates as `/app/templates/...`; Metal presets as
`src/tool_eval/server/templates/...` (run from the repo root). The plan verifies that the pinned
vLLM accepts `model` inside the `--config` file; if not, the model moves to the Compose `command`
and the README command.

### `src/tool_eval/server/templates/`

Gemma 4 and Llama 3.1 tool-call chat templates copied from vLLM's `examples/` at the pinned
version (neither the CUDA base image nor the Homebrew install ships them).

### `docker-compose.yml` (root)

- `x-vllm-base` anchor: build, image name, `ports: 8000:8000`, `ipc: host`, NVIDIA GPU reservation,
  HF cache bind mount, `HF_TOKEN`, healthcheck on `/health` (Python `urllib`, no curl in the image),
  long `start_period` for first-time weight download.
- Five services (`qwen`, `gemma`, `llama`, `qwen-mid`, `qwen-weak`), each `<<: *vllm-base`,
  own `profiles: [<name>]` and `command: ["--config", "/app/presets/cuda/<name>.yaml"]`.
- Works unchanged on Windows (Docker Desktop, WSL2 backend, NVIDIA driver on the host).

### `.dockerignore` (root)

Build context is the repo root: exclude `.venv`, `workspace/`, `logs/`, `results/`, `.git`, caches.

### Python client

- `config.py`: `llm_base_url` default `http://localhost:8000/v1`, `llm_api_key` default `EMPTY`,
  new `llm_max_tokens: int = 4096`, new `llm_enable_thinking: bool = False`.
- `llm.py`: pass `max_tokens` and
  `extra_body={"chat_template_kwargs": {"enable_thinking": settings.llm_enable_thinking}}` to
  `ChatOpenAI`; update the module docstring.

### Docs and config

- `.env.example`: `LLM_BASE_URL=http://localhost:8000/v1`, `LLM_MODEL=qwen`,
  `LLM_ENABLE_THINKING=false`, `HF_TOKEN=`, `HF_CACHE_DIR=` (optional).
- `README.md`: replace the LM Studio section with a short "Model server" section: NVIDIA
  (`docker compose --profile <preset> up -d`, Windows note), macOS (brew install +
  `vllm serve --config src/tool_eval/server/presets/metal/<preset>.yaml`), and the preset list.

## Testing

- Unit: `build_chat_model` passes `max_tokens` and the `enable_thinking` flag; settings defaults.
- Unit: preset consistency — `cuda/` and `metal/` define the same presets, `served-model-name`
  equals the file stem, parsers and template names match across platforms, referenced templates
  exist, and every Compose service points at its CUDA preset.
- Rename the "LM Studio unreachable" test message.
- Static: `docker compose config` for every profile; `uv lock` succeeds on macOS.
- Manual on macOS: at least `qwen-weak` via `vllm-metal` serves `/v1/models` with the served name
  and returns structured `tool_calls`; then a short evaluation run against it.
- Manual on the RTX 5080 box: each profile becomes healthy, `/v1/models` lists the served name, and
  one tool-calling request returns structured `tool_calls`.

## Out of scope

- Automatic model switching by the evaluation runner.
- vLLM in Docker on macOS (CPU-only, no Metal access).
- Agent-level timeout (MCP-GRANITE's 360 s) — not requested.
