## 1. Client settings and fingerprint

- [ ] 1.1 Add `llm_top_p` (0.95) and `llm_top_k` (20) to `Settings` and set the `agent_temperature` default to 0.7. Send `top_p` and `top_k` (via `extra_body`) in `llm.build_chat_model`. Tests: every request carries all three values, whatever the model.
- [ ] 1.2 Make `effective_parallel_tool_calls` return `False` by default for every preset and remove `SINGLE_TOOL_CALL_PRESETS`. Update `tests/test_llm.py` for the default and the override.
- [ ] 1.3 Add `ServerIdentity` (checkpoint from `/v1/models`, vLLM version from `/version`, sha256 of the preset file from `TOOL_EVAL_PRESET_FILE` plus its quantisation, parser and template path, and the template's sha256), falling back to `unknown` values when a source cannot be read. Tests use fake endpoints and a temporary preset file.
- [ ] 1.4 Add `top_p`, `top_k`, `llm_max_tokens`, `llm_enable_thinking` and the server identity to `config_fingerprint`, to `RunRecord` (as optional fields) and to the MLflow parameters. Tests: each value changes the hash, and old records still load.

## 2. Text limits and reading meter

- [ ] 2.1 Add the visible cut marker to `utils.text.truncate`. Tests: a long text gets the marker with the correct counts; a short text gets none.
- [ ] 2.2 Make `summarize_chunk_file` reject input longer than `chunk_size` with an `InvalidInputError` that names `split_text_into_chunks`, without calling the model. Tests: whole file rejected, chunk accepted.
- [ ] 2.3 Make `split_paper_text` write `chunks/<stem>/index.json` with the chunk offsets.
- [ ] 2.4 Add a run-scoped `ReadMeter` (ContextVar) and record intervals in `read_file`, `summarize_chunk`, the pipeline, keywords and translation. Add `cost.chars_read: dict[str, int] | None`. Tests: an L1 `read_file` of a text gives 6000; an L2 paper gives the union of its 6 chunks; repeated reads are not double-counted.

## 3. Summaries, prompts and tools

- [ ] 3.1 Add `llm_ops.synthesize` and `prompts/synthesize_summary/v3.md`. `summarize_paper` returns and stores the synthesized summary. Tests use a fake LLM: one extra call per paper, and the stored text is the synthesis.
- [ ] 3.2 Write `prompts/system/v3.md` and `prompts/evaluation/v3.md`, which describe the outcome and name no tool operation. Copy `summarize_chunk`, `extract_keywords` and `translate_text` to v3, and set the `prompt_version` default to `v3`. Test: the rendered task prompt contains none of "download", "extract", "split" or "summarize", and does state the full-text requirement.
- [ ] 3.3 Replace the `2401.00001` examples in tool descriptions with a placeholder. Test: no tool description matches `\d{4}\.\d{4,5}`.
- [ ] 3.4 Replace `append_to_file` with `write_file` at level 1, keeping the distractors. Tests: writing twice keeps only the last content; the level-1 tool list has `write_file` and no append tool.
- [ ] 3.5 Rewrite the level-3 tool description. Add `complete` and `message` to `ReviewResult`, and clear `texts/`, `chunks/`, `summaries/` and `papers/` at the start of `conduct_review`. Tests: the message on success and on shortfall; a second call leaves only its own texts.
- [ ] 3.6 Raise `TOOLSET_VERSION` to 4 and update tests that list tool names or descriptions.

## 4. Paper sources

- [ ] 4.1 Add `PaperNotFoundError(ExternalServiceError)` for PDF 404, `ArxivPaper.version`, and a retry at the latest version in `download_paper_pdf`. Tests use mocked HTTP: unversioned 404 followed by versioned 200, and 404 on both.
- [ ] 4.2 Cache missing PDFs as `<stem>.missing` in the PDF cache. Later downloads raise without an HTTP request; timeouts and 429 are not cached. Tests.
- [ ] 4.3 Add a run-scoped `NetworkMeter` counting arXiv requests and PDF downloads that miss the cache, with `cost.arxiv_requests` and `cost.pdf_downloads`. Tests: a cached run counts 0.
- [ ] 4.4 Add the `prewarm` CLI command and the `make prewarm` target (topic search with 2N candidates in the window, metadata lookup, PDF download), with a summary of what was added and what is missing. Tests with fake arXiv.

## 5. Metering and scoring

- [ ] 5.1 Count `text_tool_calls` in `RunMeter` (JSON `name` + `parameters`/`arguments`, `<tool_call>`, `<function=`). Tests: the llama-style truncated call is counted; a plain final answer is not.
- [ ] 5.2 Split tool errors into `tool_errors_environment` and `tool_errors_agent` by exception type. Tests: a missing PDF counts as environment; a bad id counts as agent; the two sum to `tool_errors`.
- [ ] 5.3 Retry scoring in `run_single` on `ExternalServiceError` (4 attempts, waits 60/180/600 s, injectable for tests) without re-running the agent. List unscored runs at the end of `run_evaluation`. Tests: outage then success gives one agent run and a real status; a persistent outage gives `error` and is listed.
- [ ] 5.4 Show the new counts in `report.py` where they belong (tool reliability: text tool calls and environment/agent errors; cost: characters read per paper, cache misses), and add them to the CSV.

## 6. Server and launcher

- [ ] 6.1 In `eval-presets.sh`: write a timestamped server log per invocation, export `TOOL_EVAL_PRESET_FILE`, send one warm-up tool request after the server is ready, and run `prewarm` once before the first preset.
- [ ] 6.2 In `presets/cuda/qwen-mid.yaml`: remove `quantization: fp8` and set `gpu-memory-utilization: 0.75`. Update the budget table in `README.md`.
- [ ] 6.3 On the RTX, start `qwen-mid` and run `make tool-check PRESET=qwen-mid`. Record the GPU memory used. If 0.75 does not fit, try 0.8, and record the outcome in design.md.

## 7. Docs, paper and presentation

- [ ] 7.1 Update `README.md`: sampling, parallel policy, prompt v3, `write_file`, synthesis step, cut marker, `prewarm`, new record fields and report columns.
- [ ] 7.2 Update `docs/overleaf/methods.tex`:
  - sampling values and that they are identical for every model;
  - one tool call per reply;
  - quantisation as served;
  - the level differences listed explicitly instead of "same capabilities";
  - "text extracted" instead of "read in full";
  - the synthesis step and the visible cut;
  - new paragraph "Protocol changes after the pilot".
- [ ] 7.3 Update `docs/overleaf/data_environment.tex` (prewarmed caches, cached missing PDFs, wording "read in full") and `docs/overleaf/experiment_plan.tex`:
  - R=3 and the run counts;
  - remove the pilot table and the conclusions drawn from it;
  - limitations: the confounds that remain.
- [ ] 7.4 Update `docs/presentations/src/progress-6-7.html` with slides on the methodology review (L3 checks met by construction, false negatives of the format checks, sampling/quantisation/parallel differences, reading depth) and on the plan (protocol fixes, task variants, outcome evaluation, final run). Regenerate the PDF.

## 8. Verification

- [ ] 8.1 `make test`, ruff and mypy clean.
- [ ] 8.2 Smoke run on the Mac: qwen-mid, all levels, 3 scenarios, REPEATS=1. Check that:
  - records carry the new fields;
  - no level-3 run calls the tool more than once without reason;
  - L2/L3 summaries are a few sentences;
  - the report shows the new columns.

  Estimate the final run time from this smoke run.
