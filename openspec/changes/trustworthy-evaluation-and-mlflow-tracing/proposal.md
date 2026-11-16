# Proposal

## Why

All 5 presets ran on all 10 scenarios × 3 levels on the RTX 5080, and traces of the `llama` and `gemma` runs were inspected (2026-11-16). They showed that a "successful" run does not prove the agent did the task.

**Unread papers count as success.**
- `llama` scored `success=True` on `level1-rag` without reading a single paper: its workspace holds only `reviews/`.
- 5 of `gemma`'s 23 successful runs cite papers it never obtained (`level1-xai`: 0 of 5).
- The task prompt only asks to "summarize each paper", and no criterion checks that the text was read.

**Harness limits, not model ability, made runs fail.**
- PDFs of 25–47 MB exceeded the 20 MB limit.
- `gemma` needs 29–33 model calls at level 1, but the limit is 30.
- `llama`'s level-2 review was cut at 4,096 output tokens and ended silently as `completed`.
- `llama` looped 29 times on `authors` sent as a Python list literal.

**Search results are often off-topic.** arXiv search is sorted by newest submission:
- "protein structure prediction" returns a paper on chiral propulsion first;
- the RAG query returns a hotel-review paper.

The checks do not judge topic relevance, so off-topic reviews score 1.0.

**Run workspaces are overwritten.** `make eval-presets` reuses one workspace directory for all presets, so later presets delete earlier presets' artifacts.

**Diagnosis needs ad-hoc scripts.** Each finding above required a throw-away tracing script. Runs need persistent, browsable traces of every model call and tool call.

## What Changes

- **BREAKING (results):** the evaluation task requires summaries from each paper's full text (prompt `v2`). A new outcome criterion `papers_read` fails any review that cites a paper whose full text the run never extracted. A checks version joins the config hash.
- **PDF size limit:** PDF downloads accept up to 100 MB through a separate PDF limit; text files keep their 20 MB limit. The error message names the paper and the limit.
- **Level-3 spare candidates:** the level-3 review tool searches spare candidates, so that it can still include the requested number of papers that it actually read.
- **Step limit:** the default rises from 30 to 100 model calls.
- **Output token limit:** the default completion-token limit rises from 4,096 to 8,192.
- **Truncated replies:** a run whose final reply was cut by that limit is recorded as `failed` with an explanatory error, instead of `completed`.
- **Python-literal arguments:** tool-argument decoding also accepts Python list and dict literals, when they decode to the declared type.
- **Per-preset workspaces:** `make eval-presets` keeps each preset's run workspaces in a directory of its own.
- **arXiv sort order:** searches are sorted by relevance instead of newest submission. The on-disk cache keeps them repeatable.
- **MLflow tracing:**
  - A self-hosted MLflow 3.18 tracking server runs in its own Docker Compose project, on `localhost:5050`.
  - When `MLFLOW_TRACKING_URI` is set, each evaluation run becomes an MLflow run with parameters, metrics and artifacts, and one trace that holds every model and tool call.
  - The criteria results are attached to the trace as assessments.
  - Run and trace IDs are stored in the results file.

## Capabilities

### New Capabilities

- `review-scoring`: what an evaluation run must achieve to count as successful. This covers the task's reading requirement and the `papers_read` criterion.
- `run-tracing`: how evaluation runs are recorded in MLflow. This covers the tracking server, what each run and trace contains, and how tracing is switched on.

### Modified Capabilities

- `agent-tool-calling`:
  - default step limit 100;
  - Python-literal argument decoding;
  - truncated final replies fail the run.
- `llm-client-config`:
  - default completion-token limit 8,192;
  - the config hash also covers the checks version.
- `paper-sources`:
  - relevance-ordered search;
  - 100 MB PDF limit with an informative error.
- `review-writing`: the level-3 review fills the requested number of read papers from spare candidates.
- `model-server`: `make eval-presets` uses a workspace directory per preset.

## Impact

- **Code:**
  - `src/tool_eval/evaluation/` (`checks.py`, `runner.py`, `models.py`, a new tracing module);
  - `src/tool_eval/services/` (`arxiv.py`, `pdf.py`, `papers.py`, `reports.py`);
  - `src/tool_eval/tools/arguments.py`;
  - `src/tool_eval/config.py`, `src/tool_eval/main.py`;
  - prompts `system/v2.md` and `evaluation/v2.md`.
- **Infrastructure:** `docker-compose.mlflow.yml`, `Makefile` (`mlflow-up/down/logs`, per-preset runs dir in `eval-presets`), `src/tool_eval/server/eval-presets.sh`.
- **Dependencies:** `mlflow-skinny==3.18.0` (client). The server image is `ghcr.io/mlflow/mlflow:v3.18.0`.
- **Results:** every run made after this change has a new config hash (prompt v2, checks version, step limit), so earlier results are not comparable or resumable.
  - Search results change with the new sort order.
  - The arXiv cache fills again for the new queries.
- **Local configuration:** `.env` values that pin the old defaults must be updated: `PROMPT_VERSION`, `AGENT_MAX_STEPS`, `LLM_MAX_TOKENS`.
