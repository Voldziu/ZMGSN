# Design

## Context

See proposal.md for the motivation. Evidence comes from the 2026-11-16 runs on the RTX 5080 (all 10 scenarios × 3 levels for `qwen`, `llama` and `gemma`), from per-call traces of `llama` and `gemma`, and from an MLflow 3.18 prototype.

Relevant current state:

- **Outcome checks.** `evaluation/checks.py::evaluate_review` checks only the review file; `CRITERIA` lists nine format and metadata checks.
  - All three levels obtain full text through `services/papers.py::extract_paper_text`, which writes `texts/<id>.txt` into the run workspace:
    - level 1: `download_pdf` → `extract_text_from_pdf`;
    - levels 2 and 3: `summarize_paper`.
  - PDFs may come from the shared PDF cache, but each run's own `papers/` and `texts/` live only in its workspace.
- **Prompts.** `load_prompt(name, settings.prompt_version)` loads both `system/<v>.md` and `evaluation/<v>.md`. Bumping the version therefore needs both files.
- **Size limit.** `services/pdf.py::download_arxiv_pdf` enforces `max_bytes` from `settings.max_file_bytes` (20 MB). The same setting limits every workspace text file.
- **Level-3 search.** `services/reports.py::conduct_review` searches exactly `review_paper_count` papers and drops any it cannot summarise.
- **arXiv sort.** `services/arxiv.py::search_arxiv` sends `sortBy=submittedDate, sortOrder=descending`.
- **Runner.** `runner.run_single` ignores the messages returned by `run_task`, so a final `AIMessage` with `finish_reason="length"` and no tool calls ends the run as `completed`.
- **Argument decoding.** `tools/arguments.py::coerce_json_string_args` decodes JSON strings only.
- **Workspace collision.** `server/eval-presets.sh` passes `--runs-dir workspace/preset-runs` for every preset, and run IDs do not include the model.
- **Prototype (MLflow).** `ghcr.io/mlflow/mlflow:v3.18.0` was run with SQLite and served artifacts, and the `mlflow-skinny==3.18.0` client was used with `mlflow.langchain.autolog()`. Our LangChain 1.4.4 and LangGraph 1.2.14 are inside its supported autolog range. Wrapping `run_single` in `mlflow.start_run` and a root `AGENT` span gave:
  - one trace per run, with 117 spans at level 1 and 38 at level 3;
  - tool-internal `CHAT_MODEL` spans nested under `TOOL` spans: 5 at level 1 and 30 at level 3, equal to `RunMeter.tool_llm_calls`;
  - per-call and per-trace token usage;
  - `ERROR` tool spans carrying the exception type and message, for both `WorkspaceError` and `ValidationError`;
  - `mlflow.sourceRun` linking each trace to its run;
  - 10 code-sourced assessments per trace (the score plus nine criteria);
  - working tag and session filters.
  - `mlflow-tracing` (the 2 MB SDK) lacks `mlflow.start_run`: it raised `AttributeError`.

## Goals / Non-Goals

**Goals:**
- A successful run means a correctly formatted review built from papers the agent read in that run.
- Harness limits (PDF size, step limit, output tokens) no longer decide outcomes for capable models.
- Every run can be inspected call by call in a local MLflow UI and compared across models and levels.

**Non-Goals:**
- Judging summary quality or topical relevance with an LLM judge (possible later with MLflow scorers).
- Replacing the results JSONL and `report`. They stay the source of the thesis numbers, and MLflow is for browsing and diagnosis.
- Authentication or remote access to the tracking server.
- Migrating earlier results into MLflow.

## Decisions

### `papers_read` = the cited paper's text was extracted in this run's workspace
- The criterion passes when `texts/<safe-stem(id)>.txt` exists in the run workspace for every arXiv ID cited in the review sections. Its detail lists the unread IDs.
- `evaluate_review` receives the run workspace explicitly, instead of deriving it from the review path.
- Extraction is the one step all levels share before any full-text summary, and it can only happen through the tools, so it is a deterministic, level-neutral proof that the full text was obtained.

*Alternatives:*
- **PDF presence:** a downloaded but unread PDF would pass.
- **A per-paper summary file:** level 1 summarises chunks without a per-paper file.
- **Trace inspection:** this would make scoring depend on MLflow.

### Prompt v2 for both system and evaluation prompts
- `evaluation/v2.md` adds that each summary must be based on the paper's full text obtained with the tools, never on search results or abstracts.
- `system/v2.md` is v1 plus one rule saying the same.
- The `prompt_version` default becomes `v2`. `CHECKS_VERSION = 2` in `checks.py` joins `config_fingerprint`.

*Alternative:* add only the criterion without changing the prompt. Rejected, because it would penalise models for a requirement the task never stated.

### Separate PDF limit of 100 MB
- New setting `max_pdf_bytes = 100_000_000`, used only by `download_arxiv_pdf`. `max_file_bytes` stays 20 MB for text files.
- The error names the paper and either the limit or the HTTP status. `Content-Length`, when present, is checked before streaming.
- The largest failed PDF seen was 46.5 MB. With 100 MB, `gemma level3-diffusion` needed 2 calls instead of 30 failing ones.

### Level-3 spare candidates
- `conduct_review` searches `2 × review_paper_count` candidates and takes them in search order until `review_paper_count` are summarised.
- `ReviewResult` reports the skipped IDs with their reasons.
- The tool never includes an unread paper.

### Relevance-sorted arXiv search
- `search_arxiv` sends `sortBy=relevance` (the descending sort order stays). The date window filter is unchanged.
- Cache keys contain the parameters, so the new queries are fetched once and then served from the cache, which keeps them identical for every model.

### Step limit 100, output tokens 8,192, truncated replies fail
- `agent_max_steps` defaults to 100. `gemma` used 29–33 calls at level 1, with prompts up to about 10k of the 32k context.
- `llm_max_tokens` defaults to 8,192. It is not higher, so that prompt plus completion still fit the 32k context in long runs.
- `run_single` inspects the last message returned by `run_task`. If it is an `AIMessage` with `finish_reason == "length"` and no tool calls, the run is recorded as `failed` with the error `final reply hit the completion-token limit (LLM_MAX_TOKENS=…)`.

### Python-literal argument decoding
- `coerce_json_string_args` tries `json.loads` first, then `ast.literal_eval`, which only evaluates literals.
- The decoded value is kept only when it is a `list` or `dict` matching the declared type, so non-literal strings still fail validation.

### Per-preset run workspaces
- `eval-presets.sh` passes `--runs-dir workspace/preset-runs/<preset>`.
- The cache location is derived from `runs_dir.parent`, which would become `workspace/preset-runs`. The caches would still be shared between presets, but the already populated `workspace/pdf_cache` and `workspace/arxiv_cache` would no longer be used. The script therefore passes `PDF_CACHE_DIR=workspace/pdf_cache` and `ARXIV_CACHE_DIR=workspace/arxiv_cache` explicitly.

### MLflow server: its own Compose project, SQLite and served artifacts
- **File and image.** `docker-compose.mlflow.yml` (`name: tool-eval-mlflow`) runs one service from `ghcr.io/mlflow/mlflow:v3.18.0` with:
  `mlflow server --backend-store-uri sqlite:////mlflow/mlflow.db --artifacts-destination /mlflow/artifacts --serve-artifacts --host 0.0.0.0 --port 5000`.
- **Storage and access.**
  - Volume `mlflow-data` is mounted at `/mlflow`.
  - Port mapping is `127.0.0.1:5050:5000`: localhost only, because the server has no authentication, and 5050 because macOS's AirPlay receiver holds 5000.
  - The healthcheck is `/health`.
- **Why a separate project.** `make docker-down PRESET=…`, which `eval-presets` runs after every preset, never touches it. It also runs on macOS, because MLflow needs no GPU.
- **Why SQLite.** Since MLflow 3.3, spans are stored in the tracking database. The artifact store holds only attachments and artifacts. One SQLite writer is enough for sequential evaluation.
- **Upgrades.** A version bump requires `mlflow db upgrade` against the volume first.
- **Make targets.** `make mlflow-up` waits for health; there are also `mlflow-down` and `mlflow-logs`.

*Alternative:* the official MLflow Compose example with PostgreSQL and RustFS S3. Rejected: three more services and credentials for no benefit at this scale.

### MLflow client: `mlflow-skinny`, opt-in via `MLFLOW_TRACKING_URI`, one run and one trace per evaluation run
- **Client package.**
  - `mlflow-skinny==3.18.0` is added to the project dependencies.
  - `mlflow-tracing` lacks the run API.
  - The full `mlflow` pulls about 1 GB (pandas, scikit-learn, the server).
  - If `uv lock` finds a conflict with the `llm` group (vLLM), `mlflow-skinny` moves to a `tracing` group declared as conflicting with `llm`.
- **New module `evaluation/tracing.py`:**
  - `tracing_enabled()`: true when `MLFLOW_TRACKING_URI` is set.
  - `start_tracing(experiment)`:
    1. checks `<uri>/health` (an error otherwise);
    2. sets the experiment (`MLFLOW_EXPERIMENT_NAME`, default `tool-eval`);
    3. calls `mlflow.langchain.autolog(silent=True)`.
  - `traced_run(...)`: a context manager around `run_single` that:
    1. opens `mlflow.start_run(run_name=f"{model}/{run_id}")` and logs params;
    2. opens a root `mlflow.start_span(name=run_id, span_type=AGENT)` with `update_current_trace(tags={model, level, scenario, repeat}, session_id=f"{model}/{scenario}")`;
    3. after the run, logs metrics, tags (status, error) and artifacts (`review.md`, `record.json`);
    4. attaches `log_feedback` for the score and each criterion, with source `CODE`/`tool_eval.evaluation.checks`;
    5. returns the MLflow run ID and trace ID.
  - `stop_tracing()`: `mlflow.flush_trace_async_logging()`, called in a `finally` block of `run_evaluation`, because asynchronous trace logging is on by default.
- **Run record.** `RunRecord` gains `mlflow_run_id` and `trace_id` (`None` when tracing is off).
- **Disabled path.** When tracing is off, nothing in `tracing.py` imports `mlflow` at module level, and the code path is a no-op, so unit tests stay offline.
- **Autolog modes.** `run_tracer_inline` is not used: our runs are synchronous, and the documented merge issue affects async use only.
- **Known issue.** MLflow issue #27258: with async logging, some traces also upload a redundant `traces.json` artifact, which is harmless.

## Risks / Trade-offs

- **[Prompt v2 and `papers_read` lower scores, `llama` most of all.]** That is intended: earlier scores overstated the work done. The config hash separates old and new results.
- **[Relevance search changes which papers each scenario uses.]** It applies equally to every model, and the cache freezes the results after the first query.
- **[More papers in level-3 candidates mean more downloads.]** Spares are fetched only when a candidate fails, and PDFs are cached.
- **[100 steps makes runaway runs longer.]** These are rare (`qwen-weak`). The context limit (32k) still ends them as `failed`.
- **[The tracking server is down mid-run.]** Async logging retries in the background. The health check covers the common case (server never started). Trace loss mid-run does not affect the results file.
- **[`mlflow-skinny` pulls FastAPI and Databricks SDK dependencies.]** They are still much lighter than full MLflow. If the lock conflicts with vLLM, a separate group isolates them.
- **[SQLite as the single writer.]** Evaluation is sequential. For parallel writers, switch to PostgreSQL, as in the official Compose example.

## Migration Plan

1. Pull the branch and run `uv sync` (adds `mlflow-skinny`).
2. Update local `.env` values that pin old defaults: `PROMPT_VERSION=v2`, `AGENT_MAX_STEPS=100`, `LLM_MAX_TOKENS=8192`. Optionally set `MLFLOW_TRACKING_URI=http://localhost:5050`.
3. Run `make mlflow-up`, then `make eval-presets …` with the tracking URI set.
4. Start new results files: earlier runs have a different config hash and are neither resumed nor comparable.
5. Remove the prototype container on the RTX host: `docker rm -f mlflow-proto && docker volume rm mlflow-proto`.
