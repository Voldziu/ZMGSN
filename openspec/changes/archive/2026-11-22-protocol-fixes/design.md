# Design

## Context

The motivation is in proposal.md, and the evidence is in `docs/methodology-review-2026-11-21.md`, where four reviewers and a verifier checked the 150-run RTX screen. Relevant current state:

- **Sampling.** `llm.build_chat_model` sends `temperature`, `seed` and `max_completion_tokens`. vLLM fills `top_p`/`top_k` from each checkpoint's `generation_config.json`:
  - gemma gets 0.95/64;
  - llama gets 0.9/–;
  - the Qwen models get no truncation.

  `Settings.agent_temperature` defaults to 0.0, but `main.py` passes 0.7.
- **Parallel calls.** `Settings.effective_parallel_tool_calls` returns `False` only for `llama` (`SINGLE_TOOL_CALL_PRESETS`).
- **Truncation.** `utils.text.truncate` is a silent slice. `read_file`, `llm_ops.summarize`, `extract_keywords` and `translate` use it with `text_max_chars=6000`. `summarize_chunk` accepts any text file.
- **Summaries.** `services.papers.summarize_paper` summarizes up to `pipeline_max_chunks=6` chunks and joins them with blank lines. Both L2 (`get_and_summarize_paper`) and L3 (`reports.conduct_review`) use it.
- **Prompts.** Prompts are versioned files `prompts/<name>/<version>.md` chosen by `Settings.prompt_version` (`v2`). `evaluation/v2.md` and `system/v2.md` prescribe "download it and extract or summarize its text".
- **Tools.** L1 writes through `append_to_file`. The `download_pdf` and `get_and_summarize_paper` descriptions show `2401.00001`. `TOOLSET_VERSION = 3` in `tools/registry.py`.
- **Caches.**
  - `services.arxiv._request` caches only successful API responses.
  - `services.papers.download_paper_pdf` caches only downloaded PDFs.
  - `services.pdf.download_arxiv_pdf` uses an unversioned URL and raises `ExternalServiceError` on any HTTP error.
- **Scoring.** `runner.run_single` turns an `ExternalServiceError` raised during scoring into `status="error"`. Such runs are re-run in full by a later invocation, or never.
- **Fingerprint.** `runner.config_fingerprint` hashes neither `llm_max_tokens`, nor thinking mode, nor anything about the server.
- **Launcher.** `server/eval-presets.sh` writes `logs/server-<preset>.log`, overwriting it on every invocation, and sends no warm-up.

## Goals / Non-Goals

**Goals**
- Every model runs under the same client-side sampling and tool-call policy.
- The served model's identity is part of the configuration.
- No level loses text without the agent knowing. The amount read per paper is measured.
- Prompts describe the outcome only. The level-3 tool says plainly that the work is done.
- The environment (arXiv, PDFs) is the same for every run, and every run of the plan is scored.

**Non-Goals**
- Changing the success metric, adding the judge, or adding deterministic fact checks. These belong to `outcome-evaluation`.
- Task variants. These belong to `task-variants`.
- Equalising reading depth across levels. The level-1 agent still chooses how much to read; this change only makes that choice visible and measured.
- A reading tool at L2/L3. Level 3 stays one-shot.
- One quantisation scheme for all models. Only `qwen-mid` changes.

## Decisions

### Sampling: one explicit triple, sent to every model
- New settings: `llm_top_p: float = 0.95` and `llm_top_k: int = 20`.
- `agent_temperature` defaults to 0.7. This matches the CLI default, which stays, and removes the misleading 0.0.
- `top_p` goes into the `ChatOpenAI` call. `top_k` is not part of the OpenAI API, so it goes into `extra_body`, where vLLM accepts it.
- All three go into `RunRecord`, the MLflow parameters and the fingerprint.
- The values are near the vendors' recommendations for instruct models without thinking (Qwen: 0.8/20, Gemma: 0.95/64, Llama: 0.9). They truncate the tail for every model, and no model gets its own vendor tuning.
- *Alternative:* the vendor-recommended values per model. Rejected, because sampling would then differ between models in the family comparison.

### Parallel calls: `False` by default for everyone
- `effective_parallel_tool_calls` returns `False` unless `LLM_PARALLEL_TOOL_CALLS` overrides it.
- `SINGLE_TOOL_CALL_PRESETS` is removed.
- Level-1 runs of Qwen will need about 30 agent calls instead of about 8. The step limit of 100 still fits.

### Served-model identity enters the fingerprint
At the start of an evaluation, the runner builds a `ServerIdentity` from three sources:
- **(a) The server.** `GET /v1/models` gives the served checkpoint (`root`) and `GET /version` gives the vLLM version.
- **(b) The preset file.** `eval-presets.sh` exports the path in `TOOL_EVAL_PRESET_FILE`. Its sha256 covers the quantisation, parser, template path and limits, and its parsed fields give the readable parameters.
- **(c) The chat template.** The sha256 of the template file the preset names, when it names one.

If the server or the preset file cannot be read, the identity holds `unknown` values. The fingerprint still differs from a run with a known identity, so the two are never treated as comparable.

The identity, `llm_max_tokens` and `llm_enable_thinking` go into `config_fingerprint`, `RunRecord` (new optional fields, `None` for old records) and the MLflow parameters.
- *Alternative:* hash the preset file only. Rejected, because the vLLM image version and the downloaded checkpoint can change while the preset file stays the same.

### Visible truncation
`utils.text.truncate` gains a marker: `\n\n[cut: showing 6000 of 66265 characters]`. The marker is appended only when text is actually cut, and every current caller keeps using `truncate`.

`papers.summarize_chunk_file` (the level-1 `summarize_chunk`) checks the input length first. Over `chunk_size`, it raises `InvalidInputError`: "This file has N characters; a chunk holds at most 3000. Split it with split_text_into_chunks and summarize the chunks." Chunk files written by `split_paper_text` are at most `chunk_size` characters by construction.

### Characters read per paper
- A run-scoped `ReadMeter` lives in a ContextVar, set by `runner.run_single` the same way the settings override is.
- `split_paper_text` writes `chunks/<stem>/index.json`, which maps each chunk file to its `[start, end)` offsets in `texts/<stem>.txt`.
- Each reading path records the interval it passed:
  - `read_file` on `texts/<stem>.txt` records `[0, kept)`;
  - `read_file` on a chunk records the chunk's interval;
  - `summarize_chunk` and the pipeline record the chunk's interval;
  - keywords and translation record what they sent.
- `RunRecord.cost.chars_read` is a `dict[str, int]` that maps each paper id to the size of the union of its intervals. Records without it report `None`.
- *Alternative:* count characters per tool call. Rejected, because rereading the same text would be counted twice.

### Synthesis step at L2/L3
- `summarize_paper` keeps summarizing up to `pipeline_max_chunks` chunks, then calls `llm_ops.synthesize(chunk_summaries, prompt=load_prompt("synthesize_summary", version))`, which returns a summary of 3 to 5 sentences.
- It stores only that summary, and `PaperSummary.summary` is that summary. `chunks_summarized` and `chunks_total` stay.
- The cost is one more LLM call per paper (5 per review), attributed to the tool like the others.

### Prompt version v3
- New files: `system/v3.md` and `evaluation/v3.md`. They describe the outcome:
  - N papers in the window;
  - each summary reflects the paper's full text, not only the abstract;
  - the exact file structure;
  - reply with the path when it is saved.

  They name no tool operation.
- `summarize_chunk/v3.md`, `extract_keywords/v3.md` and `translate_text/v3.md` are copies of v2. Every prompt name must exist in the active version.
- `synthesize_summary/v3.md` is new.
- `Settings.prompt_version` defaults to `v3`. The fingerprint already hashes both prompts.

### Tools
- **Placeholder ids.** Descriptions say: "an arXiv identifier returned by a search (format `<arxiv_id>`)".
- **`write_file(path, content)`** replaces `append_to_file` at level 1. It is built on `utils.text.write_text` with the same `.md`/`.txt` and workspace restrictions.
- **Level-3 tool.**
  - The description states that it searches the date window, reads the full text of N papers, summarizes each one, and writes `reviews/review.md` in the required structure.
  - `ReviewResult` gains `complete: bool` and `message: str`. The message is either "Review with 5 papers read in full saved to reviews/review.md. The task is complete." or "Only K of 5 papers could be read; the review is incomplete."
  - `conduct_review` first removes `texts/`, `chunks/`, `summaries/` and `papers/` in the run workspace. The shared PDF cache is untouched.
- `TOOLSET_VERSION` becomes 4.

### Caches
- **`prewarm` CLI command** (`tool-eval prewarm --scenarios … --settings …`, `make prewarm`):
  - for every scenario, runs `search_arxiv(topic, max_results=2N, window)` exactly as `conduct_review` does;
  - runs `fetch_arxiv_papers` on the candidates;
  - downloads each candidate's PDF into the PDF cache.

  It prints the number of entries added and the PDFs that are missing. `eval-presets.sh` runs it once before the first preset.
- **PDF downloads.**
  - `download_arxiv_pdf` raises a new `PaperNotFoundError(ExternalServiceError)` on 404.
  - `download_paper_pdf` catches it once and retries at the latest version. The version comes from the arXiv metadata id (`…v3`), so `ArxivPaper` gains `version: int` parsed from the entry id.
  - On a second 404 it writes `<stem>.missing` into the PDF cache, and later calls raise `PaperNotFoundError` straight from that marker.
- **Cache misses.** A run-scoped `NetworkMeter` (ContextVar) counts `arxiv_requests` and `pdf_downloads` that reach the network. Both go into `cost`.

### Scoring retries
- In `run_single`, `evaluate_review` is wrapped in up to 4 attempts, waiting 60, 180 and 600 s between them, on `ExternalServiceError` only. The arXiv client's own retries still run inside each attempt.
- `run_evaluation` ends by listing the plan's run ids that have no record with an outcome, and returns them.
- *Alternative:* store unscored runs and score them later with a separate command. Rejected for now, because in-place retries cover the observed outages (429 bursts lasting a few minutes).

### Text tool calls and error causes
- **Text tool calls.** `RunMeter._count_malformed` matches replies with no `tool_calls` whose text matches `^\s*\{\s*"name"\s*:` (followed by `parameters` or `arguments`), `<tool_call>` or `<function=`. The count goes into `cost.text_tool_calls`.
- **Error causes.** `RunMeter.on_tool_error` classifies the exception. `ExternalServiceError`, including `PaperNotFoundError`, and `httpx` errors are environment errors; everything else is an agent error. The counts go into `cost.tool_errors_environment` and `cost.tool_errors_agent`.

### Launcher and presets
- `eval-presets.sh` writes `logs/server-<preset>-<YYYYmmdd-HHMMSS>.log` and exports `TOOL_EVAL_PRESET_FILE`.
- Once the server is ready, it sends one warm-up chat request with a tool, the same as `tool-check`, before running the evaluation.
- `presets/cuda/qwen-mid.yaml`: remove `quantization: fp8` and set `gpu-memory-utilization: 0.75`.

### Paper and presentation
The paper is edited in the same change, as the loop guard was:
- `methods.tex`:
  - sampling, quantisation and the parallel policy;
  - the "same capabilities" wording becomes an explicit list of differences;
  - "read in full" becomes "text extracted";
  - the synthesis step and the visible truncation;
  - new paragraph "Protocol changes after the pilot".
- `data_environment.tex`:
  - prewarmed caches;
  - "read in full" wording;
  - prompt v3.
- `experiment_plan.tex`:
  - R=3;
  - remove the pilot table and the text that depends on it;
  - new run counts.
- `progress-6-7.html` gains slides on the review findings (L3 checks are met by construction; false negatives; sampling and quantisation) and on the change plan.

## Risks / Trade-offs

- **[The outcome-based prompt lowers L1/L2 success for strong models.]** That is what the protocol intends; the old prompt helped L1/L2. Qwen at L1 should be checked on a short smoke run before the final run.
- **[`parallel_tool_calls=false` makes Qwen L1 slower.]** It costs about 30 instead of about 8 agent calls, and the final run gets longer. A run-time estimate is made from the smoke run before starting the final run.
- **[The synthesis step can lose detail.]** The judge in `outcome-evaluation` rates faithfulness against the paper text, so a lossy synthesis would show up there.
- **[`top_k=20` might hurt one family.]** The values are reported, and they stay the same for every model.
- **[`/version` or `/v1/models` is unavailable on vllm-metal.]** The identity then falls back to `unknown` and the preset-file hash, which still separates configurations.
- **[`qwen-mid` in BF16 does not fit at 0.75.]** A task verifies startup on the RTX. If it fails, try 0.8 before falling back to FP8 and documenting the reason.

## Migration Plan

1. Implement and test locally, with a smoke run of qwen-mid on the Mac (3 scenarios, all levels).
2. Commit and push the branch, then pull on the RTX.
3. Move `results/presets-*` to `results/old/screen-1121/`.
4. Check that `qwen-mid` starts in BF16.
5. Run `make prewarm`.
6. The final run itself starts after `task-variants` is in.

Rollback: revert the commits. Old results stay readable, because every new record field is optional.

## Open Questions

- Exact top-p/top-k values (0.95/20 is assumed). The user may prefer other uniform values.
