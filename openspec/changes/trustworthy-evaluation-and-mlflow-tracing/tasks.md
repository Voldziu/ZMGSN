# Tasks

## 1. Reading requirement and `papers_read`

- [ ] 1.1 Add `system/v2.md` (v1 plus a rule to base summaries on the full text obtained with the tools) and `evaluation/v2.md` (each summary must come from the paper's full text, never from search results or abstracts). Make `v2` the `prompt_version` default. Verify with a test that the rendered v2 task states the requirement and that both v2 prompts load.
- [ ] 1.2 Add the `papers_read` criterion to `checks.py`. It passes only if every cited ID has `texts/<safe-stem(id)>.txt` in the run workspace, and its detail lists the unread IDs. Pass the workspace into `evaluate_review` from `run_single`. Verify with tests for:
  - no papers read;
  - one paper missing;
  - all read;
  - a PDF present only in the shared cache.
- [ ] 1.3 Add `CHECKS_VERSION = 2` and include it in `config_fingerprint`. Verify with a test that the fingerprint changes with the checks version.

## 2. PDFs and level-3 candidates

- [ ] 2.1 Add the `max_pdf_bytes` setting (default 100 MB) and use it in `download_paper_pdf`/`download_arxiv_pdf`; text files keep `max_file_bytes`. Check `Content-Length` before streaming. Verify with tests: a 30 MB mocked PDF downloads; a PDF over the limit fails naming the ID and limit; a 404 fails naming the ID and status.
- [ ] 2.2 In `conduct_review`, search `2 × review_paper_count` candidates and fill the review with read papers in search order, reporting skipped IDs with reasons in `ReviewResult` and never including an unread paper. Verify with tests: one failing candidate still yields 5 sections, and too few readable candidates yields only the read ones with the shortfall reported.

## 3. arXiv relevance order

- [ ] 3.1 Send `sortBy=relevance` in `search_arxiv`, keeping the date window. Verify with a test on the request parameters, and that the cache key differs from the old date-sorted query.

## 4. Agent limits and arguments

- [ ] 4.1 Change the defaults to `agent_max_steps=100` and `llm_max_tokens=8192`, and update `.env.example`. Verify with settings-default tests.
- [ ] 4.2 In `run_single`, record a run whose final message is an `AIMessage` with `finish_reason == "length"` and no tool calls as `failed`, with an error naming the token limit. Verify with a runner test using a patched agent that returns such a message, and that a normal final reply stays `completed`.
- [ ] 4.3 Extend `coerce_json_string_args` to fall back to `ast.literal_eval`, keeping the value only for a declared `array`/`object` with a matching type. Verify with tests: Python list literal decoded, Python dict literal decoded for an object argument, plain text still rejected, string arguments unchanged.

## 5. Per-preset workspaces

- [ ] 5.1 In `eval-presets.sh`, pass `--runs-dir workspace/preset-runs/<preset>` and set `PDF_CACHE_DIR=workspace/pdf_cache` and `ARXIV_CACHE_DIR=workspace/arxiv_cache` for the run. Update the README. Verify with `make eval-presets PRESETS="qwen-weak" REPEATS=1` on the Mac: the workspace appears under `workspace/preset-runs/qwen-weak/`, and the shared caches are reused.

## 6. MLflow tracking server

- [ ] 6.1 Add `docker-compose.mlflow.yml` (project `tool-eval-mlflow`) running `ghcr.io/mlflow/mlflow:v3.18.0` with the SQLite backend, served artifacts, volume `mlflow-data`, port `127.0.0.1:5050:5000` and a `/health` healthcheck. Verify that `docker compose -f docker-compose.mlflow.yml config` passes.
- [ ] 6.2 Add Makefile targets `mlflow-up` (start and wait for health), `mlflow-down` and `mlflow-logs` in a new help section, and document the server in README (start, UI URL, `mlflow db upgrade` before version bumps). Verify on the RTX host:
  - `make mlflow-up` reports healthy and `curl localhost:5050/version` returns `3.18.0`;
  - `make docker-down PRESET=qwen` leaves the MLflow container running.

## 7. MLflow tracing client

- [ ] 7.1 Add `mlflow-skinny==3.18.0` to the dependencies and update `uv.lock`. If it conflicts with the `llm` group, use a separate `tracing` group that conflicts with `llm`. Verify that `uv lock` and `uv sync` succeed on macOS, and that the CUDA image still builds.
- [ ] 7.2 Add `evaluation/tracing.py` with `tracing_enabled`, `start_tracing` (health check, experiment, `mlflow.langchain.autolog(silent=True)`), `traced_run` and `stop_tracing` (flush), importing `mlflow` lazily. Verify with unit tests:
  - with tracing disabled, nothing touches MLflow;
  - with tracing enabled and the server unhealthy, `start_tracing` raises an error naming the URI.
- [ ] 7.3 Wire tracing into `run_evaluation`/`run_single`:
  - one MLflow run per evaluation run, with params, metrics, tags and artifacts (`review.md`, `record.json`);
  - a root `AGENT` span with tags and the session `<model>/<scenario>`;
  - `log_feedback` for the score and each criterion (source `CODE`);
  - `mlflow_run_id` and `trace_id` on `RunRecord`;
  - a flush in `finally`.
  Verify with a unit test using a fake `mlflow` module that the calls, names and record fields are as specified.
- [ ] 7.4 Make `main.py` exit with code 2 and a clear message when tracing is enabled but the server is unhealthy. Document `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` in `.env.example` and README. Verify with a CLI test.

## 8. Verification

- [ ] 8.1 Run `uv run pytest`, `ruff check`, `ruff format --check` and `mypy`, and verify they all pass.
- [ ] 8.2 On the RTX host, with `make mlflow-up` and `MLFLOW_TRACKING_URI=http://localhost:5050`, run `make eval-presets PRESETS="qwen llama gemma" REPEATS=1 ONLY= LEVELS="level1 level2 level3"`. Verify that:
  - every run has one MLflow run and one trace with assessments;
  - the traces contain nested tool-internal model calls and error spans where tools failed;
  - `papers_read` is reported for every run;
  - no run fails on a PDF size limit;
  - `gemma` level 1 no longer hits the step limit;
  - per-preset workspaces are kept.
  Record success rate and score per model and level, compared with the 2026-11-16 baseline.
- [ ] 8.3 Remove the prototype container and volume on the RTX host (`mlflow-proto`). Verify that `docker ps -a` no longer lists it.

## Workflow follow-up

- Archive the change after section 8 passes. This syncs `review-scoring` and `run-tracing` and updates `agent-tool-calling`, `llm-client-config`, `paper-sources`, `review-writing` and `model-server`.
