# tool_eval

How tool granularity (`level1`, `level2`, `level3`) affects an LLM research-assistant agent.

## Setup

```sh
uv sync
cp .env.example .env
```

## Quick start: run an evaluation

1. **Model server, one-time setup.**
   - **macOS (Apple Silicon):** run `make mac-install`.
   - **Linux, or Windows with WSL 2 and an NVIDIA GPU:** meet the host prerequisites under [Model server](#model-server) (driver ≥ 580, NVIDIA Container Toolkit, `docker` group).

   You don't start a model yourself: `make eval-presets` starts and stops each one.
2. **Tracing (optional, recommended):**
   ```sh
   make mlflow-up
   export MLFLOW_TRACKING_URI=http://localhost:5050
   ```
   On a WSL host, open the UI from Windows at the same address.
3. **Run:**
   ```sh
   make eval-presets PRESETS=qwen REPEATS=1                                              # quick check
   make eval-presets PRESETS="qwen llama gemma" REPEATS=1 ONLY= LEVELS="level1 level2 level3"   # all scenarios
   ```
4. **Results:**
   - the quality table of the report is printed after each model;
   - all models together: `uv run python -m tool_eval.main report results/presets-*.jsonl --csv results/summary.csv` (see [Report](#report));
   - records go to `results/presets-<model>.jsonl`, and the log to `results/presets-<model>.log` (see [Log](#log));
   - reviews go to `workspace/preset-runs/<model>/`;
   - with tracing on, every run and trace is at `http://localhost:5050`.

Notes:
- **One model at a time,** because all presets use port 8000. Re-running a command resumes unfinished runs.
- **arXiv rate-limits by IP.** Don't run several evaluations at once from one network. On repeated `429` errors, wait and re-run; runs that ended in `error` are retried.
- **Corporate macOS (TLS-inspecting proxy):**
  - arXiv needs `SSL_CERT_FILE` pointing at the system certificates:
    ```sh
    security find-certificate -a -p /Library/Keychains/System.keychain /System/Library/Keychains/SystemRootCertificates.keychain > ~/ca-bundle.pem
    export SSL_CERT_FILE=~/ca-bundle.pem
    ```
  - `uv` must be on `PATH`.
- **macOS results are not comparable with the CUDA (RTX) runs:** they use different 4-bit checkpoints (MLX).
- **Everyone runs their own MLflow server** (`make mlflow-up`). It listens on port 5050 on all interfaces without authentication, so you can also open it from your other devices on the local network. On WSL, that also needs mirrored networking and a Windows firewall rule for port 5050.

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

A loop guard stops a run whose agent is clearly stuck and records it with status `loop`: when the agent requests the same tool call (same tool, same arguments) for the 5th time in a row (`AGENT_MAX_REPEATED_CALLS`; the call does not execute), or after 10 failed tool calls in a row (`AGENT_MAX_TOOL_ERROR_STREAK`; the model is not called again). Tool-internal model calls do not count, and `0` switches a check off. Both limits are part of the configuration hash, so results files written before the guard existed cannot be resumed; move them aside (for example to `results/old/`), `report` still reads them.

## Evaluate

Several presets in one go (starts and stops each server itself, on macOS or with Docker; no server may be running):

```sh
make eval-presets                                         # all presets, rag-2024h1, level1, 3 repeats
make eval-presets PRESETS="llama gemma" REPEATS=10 ONLY= LEVELS="level1 level2 level3"   # ONLY= runs all scenarios
```

Results go to `results/presets-<preset>.jsonl`, its log to `results/presets-<preset>.log`, and each run's workspace (review, PDFs, extracted texts) to `workspace/preset-runs/<preset>/<run_id>/`; the PDF and arXiv caches are shared in `workspace/`. Running the same command again resumes unfinished runs (delete the results file to start over). Server logs: `logs/server-<preset>.log`.

Each run draws a fresh random sampling seed, sends it with every model request of the run (including those made inside tools) and stores it as `seed` in its record. Without it, vLLM samples deterministically from server start, so repeat *n* would come out identical after every restart. The seed is not part of `config_sha256`, so repeats stay comparable.

Runs go repeat by repeat. Each block of one repeat and one scenario has a random planned order over all levels of the tool set, and its levels run in that order, so no level always runs first and pays for cold caches. The plan comes from an order seed: `--order-seed N` fixes it; without it, a resumed evaluation reuses the seed recorded in its results file, and a new file draws one.

Each record stores `order_seed` and its `block_position`, 1-based in the plan over all levels:
- Positions stay unique when a file is resumed with other `--levels`.
- A subset of the levels keeps its planned relative order, so its positions may have gaps.
- A retried run keeps its planned position; `started_at` shows when each run actually ran.
- `runner.block_order(TOOLSETS, seed=…, repeat=…, scenario_id=…)`, with `TOOLSETS` from `tool_eval.tools.registry`, recomputes a block's plan.

Like the sampling seed, the order seed is not part of `config_sha256`.

What each record measures (`cost`):
- **Model calls and tokens:** for the agent loop and for calls made inside tools (`agent_*`, `tool_*`). The total includes the tool-internal calls.
- **Tool calls and tool errors.**
- **Malformed tool calls:** agent tool calls that never start a tool. `invalid_tool_calls` have arguments that could not be parsed; `unknown_tool_calls` name a tool the level does not offer.
- **Time:** the summed duration of agent model calls, tool-internal model calls and tool calls (`agent_llm_seconds`, `tool_llm_seconds`, `tool_seconds`). Tool time minus its model calls is mostly network: arXiv, PDF downloads. Parallel tool calls all count.
- **Prompt overhead:** `first_call_input_tokens`, the input tokens of the first agent call (system prompt, task and tool schemas).
- **Wall time:** `wall_seconds` covers building the agent and the agent loop; scoring the review is not included.

Records written before a measurement existed show it as `null`, and the report leaves them out of that measurement's means.

arXiv API responses (searches and ID lookups, including the evaluation's own checks) are cached on disk in `arxiv_cache/` next to the runs directory (`workspace/arxiv_cache` by default; `ARXIV_CACHE_DIR` overrides it). Every model and repeat therefore sees the same search results, and arXiv is asked only once per query. Requests are spaced 3 s apart and retried on 429, 5xx and timeouts (up to 5 attempts, honouring `Retry-After`). Delete the directory to fetch fresh results.

Single runs against a server you started yourself:

```sh
uv run python -m tool_eval.main run                          # all scenarios, all levels, 10 repeats
uv run python -m tool_eval.main run --only rag-2024h1 --levels level1 --repeats 1   # quick check
uv run python -m tool_eval.main report                       # summary of results/results.jsonl
uv run python -m tool_eval.main report results/presets-*.jsonl --csv results/summary.csv   # all models
```

Scenarios are in `scenarios/scenarios.json`. Run options: `--scenarios`, `--only`, `--levels`, `--repeats`, `--temperature`, `--output`, `--runs-dir`, `--order-seed`.

A run is successful only if it completed (it neither hit the step limit, nor was stopped by the loop guard, nor failed) and every check passes, including `papers_read`: every paper cited in the review must have had its full text extracted in that run's workspace (`texts/<id>.txt`). Reviews written from search results or abstracts fail it. Runs that hit the step limit, ended as `loop` or failed keep their score (the fraction of passed checks) as partial credit. Each record stores the verdict as `success`; `outcome.success` only says whether every check passed. The report applies the rule to older results files too.

## Report

`report FILE...` combines results files, for example one per model, and summarises every model and level. It uses the scored runs: every run except those with an infrastructure error. It stops if a run of one model appears twice, whatever its configuration: a file passed twice, or two results files of the same model such as `smoke-qwen.jsonl` and `presets-qwen.jsonl`. The error names the run and both files. It warns when one model's runs come from different configurations. A value without data, such as the success rate of a level whose runs all ended in errors or repeatability with one repeat, is undefined (`-`), not zero. Sections:

- **Overview:** the success rate with its 95% Wilson interval, per model × level.
- **Quality:** successes, mean score, and runs that hit the step limit, were stopped by the loop guard (`loop`), failed or had errors.
- **Stability:** pass^k for k = 1..10. For each scenario it is C(c,k)/C(n,k), with c successes in n scored runs, averaged over scenarios. It is undefined above the fewest scored runs of any scenario.
- **Tool calls:** the tool-error rate (per tool call), the malformed-call rate (per attempted call: tool calls plus malformed calls) and the malformed calls per run.
- **Cost and time:** means per run of:
  - tokens, including tool-internal calls, with input and output separately;
  - tokens per success (mean tokens / success rate);
  - prompt overhead (first-call input tokens);
  - LLM calls, and model and tool time;
  - wall time.
- **Level differences:** later level minus earlier level within each model, in success rate and tokens. Each difference is the mean of the per-scenario differences over the scenarios both levels share, so every scenario weighs the same even when a level has unretried errors. Each has a 95% interval from a cluster bootstrap over scenarios (10,000 resamples, seeded, so reports are reproducible); `*` marks an interval that excludes zero.
- **Checks:** the pass rate of each check.
- **Further tables:** repeatability and consistency, per scenario, and the failed and error runs.

`--csv FILE` also writes one row per model and level with every metric, for plots. Interval bounds have their own columns, as do `pass_1` … `pass_10` and one `check_<name>` per check. Undefined values are empty cells.

## Log

`run` writes its log next to its results file, with the suffix `.log`: `--output results/presets-qwen.jsonl` logs to `results/presets-qwen.log`, so `--output` must not end in `.log` itself. A resumed evaluation appends to the same file. Only `run` writes a log file; `report`, the tests and code that imports the package do not.

- **Console:** lines from INFO up: starts, ends and warnings. `TOOL_EVAL_LOG_LEVEL=DEBUG` also shows the agent's steps live.
- **File:** every line from DEBUG up, including the steps, never rotated. Other libraries' lines appear from WARNING up.

Every line written during a run carries `[<run_id>]`, including warnings from the services that tools call (arXiv, PDF downloads) and from MLflow tracing. The run ID does not name the model, but each log belongs to one results file and therefore to one model. The evaluation and each run report their start and end:

- **Evaluation start:** `evaluation: model=…, levels=…, scenarios=…, repeats=…, order_seed=…, finished=…, to_run=…`, where `finished` counts the runs already in the results file and `to_run` those still to execute.
- **Run start:** `[<run_id>] start: position p/n, seed=…`, the run's place in its block plan and its sampling seed. It is written before the agent starts, so a run that is stopped or crashes still leaves it.
- **Run end:** `[<run_id>] end: <status>, success=…, tokens=…, wall=… s, tool_calls=…, tool_errors=…`, the same values as the run's record.

Between them, each step of the agent gets short DEBUG lines, written as soon as the step ends, so a run stopped by the step limit or by hand keeps the steps it took:

- **Model reply:** `[<run_id>] #n model -> k tool calls, … s`, adding malformed calls (unparseable or naming a tool the level does not offer), a finish reason other than a normal stop (such as `finish=length`) and the start of any text.
- **Tool call:** `[<run_id>] #n read_file(path='reviews/review.md') -> ok, 0.1 s: <result>` or `-> error, 0.4 s: InvalidInputError: <message>`, where `#n` is the reply that requested it. Model calls made inside a tool get no lines of their own; the tool's line counts them, for example `31 model calls in 1150.2 s`.
- Arguments, results and text are cut to 150 characters and kept on one line. The full messages are in the MLflow trace.

Two warnings point at a run that seems stuck in a loop: `same tool call 3 times in a row: <call>`, when the agent repeats a call with the same arguments, and `5 tool calls in a row failed; the last one: <tool>`. Each is written at most once per run. They only observe: the run goes on until the agent stops, the loop guard stops it (at the 5th identical call or the 10th failed call, with a `loop guard stopped the run: …` warning) or the step limit ends it.

## Tracing with MLflow

A self-hosted MLflow 3.18 server records every evaluation run: its settings, results and review as an MLflow run, and every model and tool call (with tokens and errors) as one trace. It runs in its own Docker Compose project (`docker-compose.mlflow.yml`), needs no GPU, and listens on all interfaces (port 5050) without authentication. Everyone runs their own server.

```sh
make mlflow-up                                   # start and wait until healthy; UI at http://localhost:5050
export MLFLOW_TRACKING_URI=http://localhost:5050 # or put it in .env; unset = no tracing
make eval-presets PRESETS="qwen llama"           # each run becomes an MLflow run with one trace
make mlflow-down                                 # stop; runs and traces stay in the mlflow-data volume
```

- Each run is logged in the experiment `MLFLOW_EXPERIMENT_NAME` (default `tool-eval`) as `<model>/<run_id>`. It carries:
  - **parameters:** model, level, scenario, repeat, seed, order seed, block position, limits, versions and config hash;
  - **metrics:** score, success (completed and every check passed), model and tool calls, tool errors, malformed tool calls, tokens (total, agent, tool-internal and the first call's input), model and tool time, and wall time. A measurement that the record does not know is not logged;
  - the review as an artifact, and one trace.
- The trace nests every agent model call, tool call and model call made inside a tool. Failed tool calls are marked as errors with the exception. The score and every criterion are attached as assessments.
- Filter traces in the UI by tags (`model`, `level`, `scenario`) or by session `<model>/<scenario>`.
- The results file stores `mlflow_run_id` and `trace_id` for each run.
- If `MLFLOW_TRACKING_URI` is set but the server is not healthy, the evaluation stops before the first run.
- Upgrading the server image: back up the volume and run `mlflow db upgrade` with the new image first; keep the image tag and `mlflow-skinny` in `pyproject.toml` on the same version.

## Checks

```sh
uv run pytest
```
