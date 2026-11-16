# Proposal

## Why

Running `qwen` on all 10 scenarios × 3 tool levels on the RTX 5080 (2026-11-16) gave 26/30 successes. Four problems came from the harness or the tool design, not from the model:

- **PDF text crash.** `level3-diffusion-2023h2` crashed with status `error`: extracted PDF text contained lone UTF-16 surrogates, and writing it raised `UnicodeEncodeError`.
- **Duplicated level-2 reviews.** Both level-2 failures (`rag`, `protein`) were whole reviews written 2–3 times into one file, giving 10 and 15 sections. The level-2 tools can only append to the review, so a rewrite stacks on top of the old text.
- **Repeats that are not independent.** vLLM samples deterministically from server start. Repeat *n* reproduced token-for-token across sessions: `qwen-weak` used 500,923/6,783 tokens on two different days, and `llama` scored 0.11/0.33/0.67 twice. Repeats collected in different sessions are therefore duplicates, not new samples.
- **arXiv rate limiting.** On 2026-11-15 arXiv answered `429` and timed out after a day of runs, which turned every `gemma` run into status `error`. A full evaluation (5 presets × 3 levels × 10 scenarios × 10 repeats) sends far more requests than that day did.

## What Changes

- **PDF text.** Lone surrogates in extracted text are replaced before the text is written, so a single odd character no longer ends a run. `fontTools` is added so `pypdf` can decode the CFF Type 1 fonts used in LaTeX papers.
- **BREAKING (level-2 tool set).** `append_to_review` is replaced by `write_review`, which overwrites `reviews/review.md` with the full review.
- **Level-2 bibliography.** `append_bibliography` writes the `## Bibliography` section and replaces any existing one instead of adding another.
- **Sampling seed.** Every run draws a fresh random seed, sends it with each model request of that run (the agent's and the tools' own), and records it in the run record. The seed is not part of the config hash, so repeats stay comparable.
- **arXiv responses.**
  - They are cached on disk and shared across runs, keyed by the request, for both searches and ID lookups (including the evaluation's own checks).
  - Requests that get `429`, a timeout or a server error are retried with increasing waits, honouring `Retry-After`.
  - The existing 3-second spacing between requests stays.

## Capabilities

### New Capabilities

- `paper-sources`: how papers are obtained, meaning arXiv metadata and search access (caching, rate limits, retries) and PDF text extraction.
- `review-writing`: how each tool level writes the review file. Covers the level-2 overwrite and bibliography semantics.

### Modified Capabilities

- `llm-client-config`: adds a per-run sampling seed that is sent with every request and recorded in results. The configuration hash also covers the version of the tool definitions, so level-2 runs made before and after this change are not mixed.

## Impact

- **Code:**
  - `src/tool_eval/services/arxiv.py` (cache, retry)
  - `src/tool_eval/services/pdf.py` (text clean-up)
  - `src/tool_eval/services/reports.py` and `src/tool_eval/tools/level2/` (write/replace semantics)
  - `src/tool_eval/config.py`, `src/tool_eval/llm.py`, `src/tool_eval/evaluation/runner.py`, `src/tool_eval/evaluation/models.py` (seed)
- **Dependencies:** `fonttools`.
- **Results:**
  - Level-2 runs made before this change use a different tool set, and are not comparable.
  - New run records carry `seed`.
  - arXiv results come from the cache once fetched, so later runs see the same search results as earlier ones.
- **Disk:** a new arXiv cache directory next to the existing PDF cache.
