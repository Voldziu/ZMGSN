# tool_eval

How tool granularity (`level1`, `level2`, `level3`) affects an LLM research-assistant agent.

## Setup

```sh
uv sync
cp .env.example .env
```

## Model server

Presets (set `LLM_MODEL` in `.env` to the preset name): `qwen`, `gemma`, `llama` (≤ 12 GB VRAM), `qwen-mid` (~8 GB), `qwen-weak` (~4 GB). Definitions: `src/tool_eval/server/presets/{cuda,metal}/`. Both paths serve `http://localhost:8000/v1`.

NVIDIA GPU — Linux, or Windows with Docker Desktop (WSL 2 backend). Host prerequisites:

- NVIDIA driver ≥ 580 (CUDA 13).
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), registered with Docker: `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
- Your user in the `docker` group: `sudo usermod -aG docker $USER`, then log in again.
- Check: `docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi` lists the GPU.

```sh
docker compose --profile qwen up -d --build
docker compose --profile qwen down   # before starting another preset (all use port 8000)
```

macOS (Apple Silicon, GPU via vllm-metal; Docker on macOS has no GPU access):

```sh
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
# without a current Xcode: curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
#                          (installs into ~/.venv-vllm-metal; use ~/.venv-vllm-metal/bin/vllm)
GLOO_SOCKET_IFNAME=lo0 vllm serve --config src/tool_eval/server/presets/metal/qwen.yaml   # from the repo root
```

`GLOO_SOCKET_IFNAME=lo0` avoids a long startup stall when macOS cannot resolve the local hostname.

Gated models need `HF_TOKEN` (in `.env` for Docker, exported in the shell for macOS). Mac presets use MLX 4-bit weights, so their results are not directly comparable with the CUDA runs.

Per-preset client behaviour: `llama` gets one tool call per reply (`parallel_tool_calls: false`), because its chat template rejects several tool calls in one message; the other presets keep parallel calls. `LLM_PARALLEL_TOOL_CALLS=true|false` overrides this for any preset. Runs allow 100 model calls (`AGENT_MAX_STEPS`) and 8192 output tokens per reply (`LLM_MAX_TOKENS`), enough for models that call one tool per turn and for writing a whole review in one call.

## Evaluate

Several presets in one go (starts and stops each server itself, on macOS or with Docker; no server may be running):

```sh
make eval-presets                                         # all presets, rag-2024h1, level1, 3 repeats
make eval-presets PRESETS="llama gemma" REPEATS=10 ONLY= LEVELS="level1 level2 level3"   # ONLY= runs all scenarios
```

Results go to `results/presets-<preset>.jsonl` and each run's workspace (review, PDFs, extracted texts) to `workspace/preset-runs/<preset>/<run_id>/`; the PDF and arXiv caches are shared in `workspace/`. Running the same command again resumes unfinished runs (delete the results file to start over). Server logs: `logs/server-<preset>.log`.

Each run draws a fresh random sampling seed, sends it with every model request of the run (including those made inside tools) and stores it as `seed` in its record. Without it, vLLM samples deterministically from server start, so repeat *n* would come out identical after every restart. The seed is not part of `config_sha256`, so repeats stay comparable.

arXiv API responses (searches and ID lookups, including the evaluation's own checks) are cached on disk in `arxiv_cache/` next to the runs directory (`workspace/arxiv_cache` by default; `ARXIV_CACHE_DIR` overrides it). Every model and repeat therefore sees the same search results, and arXiv is asked only once per query. Requests are spaced 3 s apart and retried on 429, 5xx and timeouts (up to 5 attempts, honouring `Retry-After`). Delete the directory to fetch fresh results.

Single runs against a server you started yourself:

```sh
uv run python -m tool_eval.main run                          # all scenarios, all levels, 10 repeats
uv run python -m tool_eval.main run --only rag-2024h1 --levels level1 --repeats 1   # quick check
uv run python -m tool_eval.main report                       # summary of results/results.jsonl
```

Scenarios are in `scenarios/scenarios.json`. Run options: `--scenarios`, `--only`, `--levels`, `--repeats`, `--temperature`, `--output`, `--runs-dir`.

A run is successful only if every check passes, including `papers_read`: every paper cited in the review must have had its full text extracted in that run's workspace (`texts/<id>.txt`). Reviews written from search results or abstracts fail it.

## Tracing with MLflow

A self-hosted MLflow 3.18 server records every evaluation run: its settings, results and review as an MLflow run, and every model and tool call (with tokens and errors) as one trace. It runs in its own Docker Compose project (`docker-compose.mlflow.yml`), needs no GPU, and listens on localhost only.

```sh
make mlflow-up                                   # start and wait until healthy; UI at http://localhost:5050
export MLFLOW_TRACKING_URI=http://localhost:5050 # or put it in .env; unset = no tracing
make eval-presets PRESETS="qwen llama"           # each run becomes an MLflow run with one trace
make mlflow-down                                 # stop; runs and traces stay in the mlflow-data volume
```

- Each run is logged in the experiment `MLFLOW_EXPERIMENT_NAME` (default `tool-eval`) as `<model>/<run_id>`. It carries parameters (model, level, scenario, repeat, seed, limits, versions, config hash), metrics (score, success, calls, tool errors, tokens, wall time), the review as an artifact, and one trace.
- The trace nests every agent model call, tool call and model call made inside a tool. Failed tool calls are marked as errors with the exception. The score and every criterion are attached as assessments.
- Filter traces in the UI by tags (`model`, `level`, `scenario`) or by session `<model>/<scenario>`.
- The results file stores `mlflow_run_id` and `trace_id` for each run.
- If `MLFLOW_TRACKING_URI` is set but the server is not healthy, the evaluation stops before the first run.
- Upgrading the server image: back up the volume and run `mlflow db upgrade` with the new image first; keep the image tag and `mlflow-skinny` in `pyproject.toml` on the same version.

## Checks

```sh
uv run pytest
```
