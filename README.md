# tool_eval

How tool granularity (`level1`, `level2`, `level3`) affects an LLM research-assistant agent.

## Setup

```sh
uv sync
cp .env.example .env
```

## Model server

Presets (set `LLM_MODEL` in `.env` to the preset name): `qwen`, `gemma`, `llama` (≤ 12 GB VRAM), `qwen-mid` (~6 GB), `qwen-weak` (~4 GB). Definitions: `src/tool_eval/server/presets/{cuda,metal}/`. Both paths serve `http://localhost:8000/v1`.

NVIDIA GPU — Linux, or Windows with Docker Desktop (WSL 2 backend); driver ≥ 580:

```sh
docker compose --profile qwen up -d --build
```

macOS (Apple Silicon, GPU via vllm-metal; Docker on macOS has no GPU access):

```sh
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
vllm serve --config src/tool_eval/server/presets/metal/qwen.yaml   # run from the repo root
```

Gated models need `HF_TOKEN` (in `.env` for Docker, exported in the shell for macOS). Mac presets use MLX 4-bit weights, so their results are not directly comparable with the CUDA runs.

## Evaluate

```sh
uv run python -m tool_eval.main run                          # all scenarios, all levels, 10 repeats
uv run python -m tool_eval.main run --only rag-2024h1 --levels level1 --repeats 1   # quick check
uv run python -m tool_eval.main report                       # summary of results/results.jsonl
```

Scenarios are in `scenarios/scenarios.json`. Run options: `--scenarios`, `--only`, `--levels`, `--repeats`, `--temperature`, `--output`, `--runs-dir`.

## Checks

```sh
uv run pytest
```
