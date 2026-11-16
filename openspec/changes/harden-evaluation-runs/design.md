# Design

## Context

See proposal.md for the motivation. Relevant current state:

- **arXiv access.** All arXiv calls go through `services/arxiv.py::_request`: tool searches (`search_arxiv`), ID lookups (`fetch_arxiv_papers`, used by level-2 bibliography and level-1 citation tools) and the evaluation's own checks (`evaluation/checks.py::ArxivLookup`, which only caches in memory).
  - `_request` already spaces requests 3 s apart within one process (`_throttle`).
  - Any `httpx.HTTPError` becomes `ExternalServiceError` at once, with no retry.
- **PDF text.**
  - `services/pdf.py::extract_pdf_text` returns `pypdf`'s text as is.
  - `services/papers.py::extract_paper_text` writes it with `utils/text.py::write_text`.
  - The size check in `write_text` encodes the text to UTF-8 and raises `UnicodeEncodeError` on lone surrogates. That error is not a `ToolEvalError`, so it escapes `ToolErrorMiddleware` and ends the run as `error`.
- **Level-2 tools.** The level-2 set is `search_papers`, `get_and_summarize_paper`, `append_to_review` and `append_bibliography`. Both writers append through `utils/text.py::append_text`. No level-2 tool reads or replaces the review.
- **Model client.**
  - `llm.py::build_chat_model` builds the `ChatOpenAI` client from `Settings`.
  - `runner.py::run_single` builds per-run settings (`settings.model_copy(update=…)`) and activates them with `use_settings`.
  - Tool-internal model calls use `get_chat_model()`, which reads the same active settings.
  - `ChatOpenAI` has a `seed` field, which vLLM honours per request.

## Goals / Non-Goals

**Goals:**
- No run ends with status `error` because of PDF text encoding or a transient arXiv rate limit.
- Level-2 rewrites replace text instead of stacking.
- Repeats are independent samples regardless of server restarts, and each run's seed is known.

**Non-Goals:**
- Changing level-1 or level-3 tools.
- Making a single run bit-for-bit reproducible across vLLM versions or GPUs. The seed is recorded, but reproduction is not promised.
- Cache invalidation or expiry. arXiv metadata for a past date window is stable.
- Per-run traces of tool calls (a separate change).

## Decisions

### Clean up text inside `extract_pdf_text`
- Replace unencodable characters with `text.encode("utf-8", "replace").decode("utf-8")`, which turns lone surrogates into `?`.
- Do this where the text is produced, so every consumer (files, chunks, summaries, level-3 pipeline) gets valid text.
- Add `fonttools` as a dependency so `pypdf` decodes CFF Type 1 fonts properly. That should also stop most garbled characters at the source.

*Alternatives:*
- Making `write_text` tolerant: hides the problem from other consumers.
- Treating the error as a tool error: the model still loses the paper.

### Level-2 tools: `write_review` overwrites, the bibliography section is replaced
- `reports.write_review(markdown)` replaces `reviews/review.md`. The level-2 tool `write_review` replaces `append_to_review` in the level-2 set, and its description says it overwrites the whole review.
- `append_bibliography` keeps its name and purpose, but `reports.append_bibliography` now replaces an existing `## Bibliography` section through the end of the file. If there is no such section, it adds one at the end.
- Assumption: the bibliography is the last section, as the evaluation prompt requires.
- `reports.append_to_review` is removed. Its only caller besides the level-2 tool is the old `append_bibliography`, which now replaces the section instead. Level 3 already writes the whole review at once with `write_text`. Tests switch to `write_review`.

*Alternatives:*
- Adding a `read_review` tool: rejected by the user in favour of overwrite.
- Folding the bibliography into `write_review`: would force the agent to supply IDs together with text, a bigger change to the tool's shape.

### Tool-definition version in the config hash
- Add `TOOLSET_VERSION` (an integer constant in `tools/registry.py`) to `config_fingerprint`. This change bumps it to 2.
- An explicit constant is simpler and more readable than hashing tool schemas, and the PR that changes tools is where someone decides comparability anyway.

*Alternative:* hash the names and descriptions of all tools. That is automatic, but any wording fix in a docstring would split results.

### Random seed per run, carried in settings
- `Settings.llm_seed: int | None = None`.
- `run_single` sets `llm_seed=secrets.randbelow(2**31)` in the per-run `model_copy`.
- `build_chat_model` passes `seed=settings.llm_seed`, so the agent model and every `get_chat_model()` inside tools share it.
- `RunRecord.seed: int | None` stores it.
- `config_fingerprint` ignores it, because the seed is a sampling draw, not a configuration.
- Interactive use outside the runner sends no seed and keeps vLLM's default behaviour.

*Alternative:* a deterministic seed derived from the run ID. The user chose fresh random seeds; recording them keeps runs auditable.

### arXiv cache and retry inside `_request`
- **Cache.**
  - Key: SHA-256 of the sorted request parameters.
  - Value: the raw Atom XML, stored as `<cache_dir>/<key>.xml` and written atomically (temp file plus rename).
  - Lookup happens before throttling, so cache hits are instant. Parsing stays the same for cached and fresh responses.
- **Cache directory.**
  - `Settings.arxiv_cache_dir: Path | None`; `None` disables the cache.
  - The runner defaults it to `runs_dir.parent / "arxiv_cache"`, mirroring `pdf_cache_dir`.
  - `_request` reads it from `get_settings()`, so tools need no new parameters.
  - `ArxivLookup` in the evaluation receives the cache directory explicitly from `main.py`.
- **Retry.**
  - Up to 5 attempts.
  - Waits: `Retry-After` if present, otherwise 5, 10, 20 and 40 s, plus up to 1 s of jitter.
  - Retried: HTTP 429, 5xx and timeout or connection errors. Other 4xx responses fail at once.
  - Every attempt still goes through `_throttle`, so the 3 s spacing holds.
- **Success only.** A response is cached only when it parses without an arXiv error entry.

*Alternative:* `hishel` or another HTTP cache library. That adds a dependency for one endpoint, and its cache keys would include headers.

## Risks / Trade-offs

- [A cached arXiv answer goes stale (for example, a paper is withdrawn).] → Accepted. Results become more comparable across models and runs, and deleting the cache directory resets it.
- [Retries make a run slower during an outage (up to about 75 s per request).] → Better than an `error` run. The step limit and the report's wall-time column make it visible.
- [The level-2 tool set changes, so earlier level-2 results are not comparable.] → `TOOLSET_VERSION` changes the config hash, so mixed runs are told apart and are not resumed.
- [`?` replacements alter a few characters in paper text.] → Negligible for summaries. `fonttools` should make them rare.
- [vLLM could ignore `seed` for some sampling paths.] → Verification checks that repeat 1 differs across two server sessions.

## Migration Plan

1. Merge and run `uv sync` (adds `fonttools`). Rebuild the CUDA image only if server dependencies change (they do not).
2. Start new results files. Old level-2 results have a different config hash and are not resumed.
3. Optionally pre-warm the arXiv cache with one run per scenario before a full evaluation.
