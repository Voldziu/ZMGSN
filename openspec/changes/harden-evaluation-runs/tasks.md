# Tasks

## 1. PDF text extraction

- [x] 1.1 In `services/pdf.py::extract_pdf_text`, replace characters that cannot be encoded in UTF-8 (lone surrogates) before returning the text. Verify with a unit test that patches the PDF reader to return text containing `"\ud835"` and checks that `extract_paper_text` writes a file containing `?`.
- [x] 1.2 Add `fonttools` to the project dependencies and update `uv.lock`. Verify that `uv sync` succeeds and that extracting a LaTeX (Computer Modern) arXiv PDF no longer logs the `pypdf` "fontTools is required" warning.

## 2. Level-2 review writing

- [x] 2.1 Add `reports.write_review(markdown)` that replaces `reviews/review.md`. Replace the level-2 tool `append_to_review` with `write_review`, with a description that says it overwrites the whole review. Verify with a test that two calls leave only the second text.
- [x] 2.2 Change `reports.append_bibliography` to replace an existing `## Bibliography` section (from its heading to the end of the file) or add one, leaving earlier text unchanged. Verify with tests: two calls give one section listing the second call's IDs, and `write_review` afterwards removes it.
- [x] 2.3 Remove `reports.append_to_review`, update the level-tool parity test (`test_every_level_can_search_summarize_and_write`) and any test that used it. Verify that `uv run pytest` passes.
- [x] 2.4 Add `TOOLSET_VERSION = 2` in `tools/registry.py` and include it in `config_fingerprint`. Verify with a test that the fingerprint changes with the version.

## 3. Per-run sampling seed

- [x] 3.1 Add `Settings.llm_seed: int | None` and pass it as `seed` in `build_chat_model`. Verify with unit tests that the client has the seed when it is set and none when it is unset.
- [x] 3.2 In `run_single`, draw `secrets.randbelow(2**31)` into the per-run settings and store it in a new `RunRecord.seed` field. Keep it out of `config_fingerprint`. Verify with tests that two runs get different seeds that are recorded, that the fingerprint is equal for them, and that a tool-internal `get_chat_model()` inside the run sees the same seed.
- [x] 3.3 Document the seed in README (results section). Verify that the README describes where the seed is recorded.

## 4. arXiv cache and retry

- [x] 4.1 Add `Settings.arxiv_cache_dir: Path | None`. Have the runner default it to `runs_dir.parent / "arxiv_cache"` and pass it to `ArxivLookup` from `main.py`. Verify with a test that a run's settings carry the cache directory.
- [x] 4.2 In `services/arxiv.py::_request`, read and write cached Atom XML keyed by a hash of the sorted parameters, using atomic writes and caching only responses that parse without an error entry. Verify with tests: a second identical request makes no HTTP call, a failed request writes no cache file, and a new process (fresh module state) reads an existing cache file.
- [x] 4.3 Retry 429, 5xx and timeout or connection errors up to 5 attempts, waiting `Retry-After` or 5/10/20/40 s plus jitter, keep the 3 s spacing, and fail at once on other 4xx responses. Verify with tests using a mocked transport and a patched sleep: 429 then 200 succeeds and waits at least `Retry-After`, 429 five times raises `ExternalServiceError`, and 400 is not retried.
- [x] 4.4 Document the arXiv cache directory and how to reset it in README. Verify that the README matches the setting name and default.

## 5. Verification

- [x] 5.1 Run `uv run pytest`, `ruff check`, `ruff format --check` and `mypy`, and verify they all pass.
- [ ] 5.2 On the RTX 5080, run `make eval-presets PRESETS=qwen REPEATS=1 ONLY= LEVELS="level1 level2 level3"` (all 10 scenarios). Verify that no run ends with status `error`, that every level-2 review has exactly 5 paper sections, and that records carry a `seed`.
- [ ] 5.3 Run `make eval-presets PRESETS=qwen-weak REPEATS=1` twice, restarting the server in between. Verify that the two repeat-1 runs have different seeds and different token counts, unlike the identical 500,923/6,783 seen before.

## Workflow follow-up

- Archive the change after section 5 passes. This syncs `paper-sources`, `review-writing` and `llm-client-config`.
