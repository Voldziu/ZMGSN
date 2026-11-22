# Proposal

## Why

A methodology review of the RTX screen (150 runs, 2026-11-21; findings in `docs/methodology-review-2026-11-21.md`) showed several protocol issues. Each one makes the comparison between levels and between models measure something other than tool granularity:
- Models sample with different hidden top-p/top-k values, and the parallel-call policy differs by model.
- Qwen 4B gets a different precision than its neighbours in the scale series.
- L1 cuts papers to 6,000 characters without saying so, while L2/L3 paste six chunk summaries together.
- The task prompt prescribes the L1/L2 process, so the L3 agent re-runs the finished macro-tool.
- L1 can only append, which duplicates sections.
- Docstring example ids are copied as if they were real papers.
- Caches warm up during the run, PDF 404s hit levels differently, scoring failures drop runs, and the configuration hash misses settings that change results.

All of this must be fixed before the final run. That run will be repeated 3 times, with base tasks and variants, on a new hash.

## What Changes

- **Sampling.** The client sends the same explicit `temperature`, `top_p` and `top_k` to every model. They are recorded in the run record, the MLflow parameters and the configuration hash.
- **BREAKING (results): one parallel-call policy for every model.** Every model gets one tool call per reply (`parallel_tool_calls=false`), not only `llama`.
- **Qwen 4B precision.** `qwen-mid` is served in BF16 instead of online FP8, with its GPU budget raised from 8 GB to 12 GB.
- **No silent truncation.**
  - When a tool shortens text for the model (`read_file`, `summarize_chunk`, keywords, translation), it appends a visible marker that gives the characters kept and the total.
  - `summarize_chunk` refuses an input longer than one chunk with a tool error that tells the agent to split the text first.
- **Characters read per paper.** Each run records how many characters of each paper's text were passed to a model.
- **One summary per paper at L2/L3.** `summarize_paper` adds a synthesis call that turns the chunk summaries into one summary of a few sentences, instead of concatenating them.
- **BREAKING (results): outcome-based prompts (version v3).** The task and system prompts state what the review must contain, not which tool steps to take. The level-3 tool description and its result say what it did: N papers read in full, the review saved in the required structure, the task complete. Level 3 stays one-shot, so neither level 2 nor level 3 gets a reading tool.
- **Level-1 writing.** Level 1 gets a `write_file` tool that replaces a file's content, instead of append-only writing. The distractor tools stay.
- **Placeholder ids.** Tool descriptions use a placeholder instead of a real-looking arXiv id (`2401.00001`).
- **Frozen caches.**
  - A `prewarm` command fills the arXiv response cache and the PDF cache for every scenario.
  - PDF download failures (HTTP 404) are cached as well, so a missing PDF behaves the same in every run.
  - A missing unversioned PDF is retried at its latest version.
  - Each run records how many arXiv requests and PDF downloads missed the cache. Agents can search with any query they choose, so the network cannot be closed.
- **Scoring retries.** When scoring fails because of an external service, only scoring is retried (with back-off), not the agent run. A run is recorded as `error` only after the retries fail. The evaluation reports cells that are still incomplete.
- **New counts.**
  - Replies that carry a tool call as plain text (the server's parser failed) are counted as `text_tool_calls`.
  - Tool errors are split into environment errors (network, HTTP 4xx/5xx, missing PDF) and agent errors.
- **Warm-up.** `eval-presets.sh` sends one warm-up request to each model before its first run.
- **Server logs.** The server log of each invocation gets its own timestamped file, and none is overwritten.
- **Clean level-3 workspace.** A repeated level-3 call starts from a clean `texts/`, `chunks/` and `summaries/`, so files from an earlier call do not accumulate.
- **BREAKING (results): wider configuration hash.** The hash also covers `llm_max_tokens`, thinking mode, sampling, and the served model identity: checkpoint, quantisation, tool parser, chat-template hash and vLLM version.
- **Paper and presentation.**
  - `methods.tex`, `data_environment.tex` and `experiment_plan.tex` describe the new protocol and gain a "Protocol changes after the pilot" paragraph. The pilot table is removed, and the incorrect "same capabilities", "read in full" and "quantised where needed" wording is fixed. The plan says R=3.
  - `docs/presentations/src/progress-6-7.html` gets the findings of the methodology review and the plan.

The loop guard keeps its default of 8 identical calls and 10 failed calls.

## Capabilities

### New Capabilities
- `tool-text-limits`: how tools shorten text for the model. Cuts are visible, chunk-sized inputs are enforced, and the characters read are counted.

### Modified Capabilities
- `llm-client-config`:
  - explicit identical sampling;
  - one tool call per reply for every preset;
  - the configuration hash covers sampling, the token limit, thinking mode and the served model identity.
- `model-server`:
  - `qwen-mid` in BF16 with a 12 GB budget;
  - a warm-up request;
  - one server log per invocation.
- `agent-tool-calling`:
  - outcome-based prompts;
  - placeholder ids in tool descriptions;
  - an overwriting write tool at level 1.
- `review-scoring`:
  - the task states the full-text requirement as an outcome, not as tool steps;
  - scoring retries external failures.
- `review-writing`:
  - one synthesised summary per paper at levels 2/3;
  - the level-3 result reports completion;
  - a repeated level-3 call starts clean.
- `paper-sources`:
  - prewarm and frozen caches;
  - cached PDF failures.
- `run-metering`:
  - text tool calls;
  - environment and agent tool errors;
  - characters read per paper.
- `run-tracing`: the MLflow parameters include the sampling settings and the served model identity.

## Impact

- **Code:**
  - `config.py`, `llm.py`;
  - `prompts/{system,evaluation,summarize_chunk}/v3.md` and a new synthesis prompt;
  - `tools/level1/*` (a new `write_file` replaces `append_to_file`), `tools/level2/papers.py`, `tools/level3/review.py`;
  - `services/papers.py`, `services/reports.py`, `services/arxiv.py`, `services/llm_ops.py`, `utils/text.py`;
  - `evaluation/runner.py`, `evaluation/cost.py`, `evaluation/models.py`, `evaluation/tracing.py`, `evaluation/report.py`;
  - `main.py` (the `prewarm` command), `server/eval-presets.sh`, `server/presets/cuda/qwen-mid.yaml`.
- **Results:**
  - A new configuration hash (TOOLSET_VERSION and prompt v3), so screen results cannot be resumed.
  - The screen stays in `results/old/` as diagnostic data only.
- **Run time:** level 1 gets slower for Qwen, which previously made parallel calls. L2/L3 gain one synthesis call per paper.
- **Dependencies:** none.
- **Docs:** `README.md`, `docs/overleaf/*.tex`, `docs/presentations/src/progress-6-7.html`.
