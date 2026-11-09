# vLLM server instead of LM Studio

## Goal

Replace LM Studio with a vLLM server run in Docker Compose on an RTX 5080 (16 GB, max 12 GB used).
One model runs at a time, selected with a Compose profile. The Python client stays
OpenAI-compatible, so on macOS any OpenAI-compatible endpoint (LM Studio, remote vLLM) still works
by changing `LLM_BASE_URL`.

## Decisions

| Topic | Decision |
| --- | --- |
| Model switching | One Compose service per preset, each with its own profile (`docker compose --profile qwen up`) |
| Preset location | Directly in `docker-compose.yml` (no separate preset file to keep in sync) |
| Compose location | Repo root; build context `.`, Dockerfile in `src/tool_eval/server/` |
| Base image | `nvidia/cuda:12.9.x-devel-ubuntu24.04` + uv; vLLM installed from the `llm` dependency group |
| Weights | Bind mount host `~/.cache/huggingface` → `/root/.cache/huggingface`; `HF_TOKEN` from `.env` |
| Model on the client | Explicit `LLM_MODEL` in `.env`, equal to the preset's `--served-model-name` |
| macOS | No platform branches; any OpenAI-compatible endpoint via `LLM_BASE_URL` |
| Thinking mode | `LLM_ENABLE_THINKING` setting, default `false`, sent per request as `chat_template_kwargs` |
| Agent controls | Aligned to MCP-GRANITE (arXiv:2609.24161): temperature 0.0, max 10 steps, `max_tokens` 4096 |

## Presets

Budget = `--gpu-memory-utilization` × 16 GB. All presets: `--max-model-len 16384`,
`--enable-auto-tool-choice`, multimodal inputs disabled.

| Profile / served name | HF repo | Quantization | Budget (util) | Tool parser | Extra flags |
| --- | --- | --- | --- | --- | --- |
| `qwen` | `cyankiwi/Qwen3.5-9B-AWQ-4bit` | AWQ 4-bit (compressed-tensors) | 12 GB (0.75) | `qwen3_coder` | `--reasoning-parser qwen3 --language-model-only` |
| `gemma` | `google/gemma-4-12B-it-qat-w4a16-ct` | QAT W4A16 | 12 GB (0.75) | `gemma4` | `--reasoning-parser gemma4 --chat-template /app/templates/tool_chat_template_gemma4.jinja --limit-mm-per-prompt '{"image":0,"audio":0}'` |
| `llama` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | AWQ 4-bit | 12 GB (0.75) | `llama3_json` | `--chat-template /app/templates/tool_chat_template_llama3.1_json.jinja` |
| `qwen-mid` | `Qwen/Qwen3.5-4B` | FP8 online (`--quantization fp8`) | 6 GB (0.375) | `qwen3_coder` | as `qwen` |
| `qwen-weak` | `Qwen/Qwen3.5-0.8B` | BF16 | 4 GB (0.25) | `qwen3_coder` | as `qwen` |

Notes:

- Llama 3.1-8B is the newest Llama that fits; Meta has released no smaller model since (Llama 4 starts above 100B total parameters).
- Qwen3.6/3.8 start at 27B, so all Qwen presets use Qwen3.5.
- If `qwen-mid` does not fit in 6 GB with FP8, fall back to `sanskar003/Qwen3.5-4B-AWQ`.
- Utilization fractions assume a 16 GB card; on another GPU they must be recomputed.

## Components

### `pyproject.toml` — `llm` dependency group

- `vllm==<pinned>` with marker `sys_platform == 'linux'` so `uv lock`/`uv sync` keep working on macOS.
- `torch` resolved from the PyTorch CUDA index matching vLLM's wheels (cu129) via `[[tool.uv.index]]`
  (explicit) + `[tool.uv.sources]`, also Linux-only.
- Not part of default groups; `uv sync` on macOS installs nothing from it.

### `src/tool_eval/server/Dockerfile`

- `FROM nvidia/cuda:12.9.x-devel-ubuntu24.04` (devel: `nvcc` needed for FlashInfer JIT on `sm_120`).
- uv copied from `ghcr.io/astral-sh/uv`, Python 3.12 installed by uv.
- `COPY pyproject.toml uv.lock` → `uv sync --frozen --only-group llm --no-install-project`.
- `COPY src/tool_eval/server/templates /app/templates`.
- `ENTRYPOINT ["vllm", "serve"]`, with `--host 0.0.0.0 --port 8000` as fixed entrypoint args.

### `src/tool_eval/server/templates/`

Gemma 4 and Llama 3.1 tool-call chat templates copied from vLLM's `examples/` at the pinned
version (the CUDA base image does not ship them).

### `docker-compose.yml` (root)

- `x-vllm-base` anchor: build, image name, `ports: 8000:8000`, `ipc: host`, NVIDIA GPU reservation,
  HF cache bind mount, `HF_TOKEN`, healthcheck on `/health` (Python `urllib`, no curl in the image),
  long `start_period` for first-time weight download.
- Five services (`qwen`, `gemma`, `llama`, `qwen-mid`, `qwen-weak`), each `<<: *vllm-base`,
  own `profiles: [<name>]` and a `command` list with model + preset flags from the table above.

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
  `LLM_ENABLE_THINKING=false`, `HF_TOKEN=`.
- `README.md`: replace the LM Studio section with `docker compose --profile <preset> up -d`,
  the preset table, and the macOS note.

## Testing

- Unit: `build_chat_model` passes `max_tokens` and the `enable_thinking` flag; settings defaults.
- Rename the "LM Studio unreachable" test message.
- Static: `docker compose config` for every profile; `uv lock` succeeds on macOS.
- Manual on the RTX 5080 box: each profile becomes healthy, `/v1/models` lists the served name, and
  one tool-calling request returns structured `tool_calls`.

## Out of scope

- Automatic model switching by the evaluation runner.
- CPU/macOS vLLM image.
- Agent-level timeout (MCP-GRANITE's 360 s) — not requested.
