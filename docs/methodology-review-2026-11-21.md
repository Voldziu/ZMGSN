# Methodology review: RTX screen, 2026-11-21

**Input data.** The diagnostic screen on the RTX:
- 5 models × 10 scenarios × 3 levels × 1 repeat, 150 runs.
- Remote commit `1f405d2`, loop guard `max_repeated_calls=5`.
- The screen is diagnostic only; no number from it goes into the paper.

**How the review was done.** Four independent reviewers (checks validity, level fairness, model fairness, statistics) and a verifier who tried to refute 15 key claims.
- 13 claims were confirmed.
- 2 were partly confirmed: C5 is about wording and where in the text it appears; C9 concerns the parse error, which ended as `failed` and was therefore not lost.

The list below combines those findings with the decisions from the grilling session. Code references point to the local checkout.

## Screen results

Successes per level:

| model | L1 | L2 | L3 |
|---|---|---|---|
| qwen 9B | 10/10 | 10/10 | 10/10 |
| gemma 12B | 8/10 | 9/10 | 10/10 |
| llama 3.1 8B | 0/9 | 1/9 | 10/10 |
| qwen 4B | 5/10 | 10/10 | 9/10 (1 loop) |
| qwen 0.8B | 0/10 | 0/10 | 1/10 (9 loop, every one with score 1.0) |

## A. Main finding: L3 success is produced by the code

- **Confirmed.** `services/reports.py` `conduct_review` writes the headings from arXiv titles, the APA citations, the bibliography and `texts/`. It also searches within the date window.
- **Result.** All 50 L3 runs scored 1.0, and the paper set was identical across all 5 models in 9 of 10 scenarios, because every model passed the topic as the query.
- **Off-topic example.** One review scored 1.0 while being off topic: `qwen-weak level3-diffusion`, query `'arXiv 2310'`, sections on mathematics and QCD.
- **What the checks measure.** They measure format and provenance. They do not measure relevance or faithfulness.

## B. Already decided (grilling), now confirmed by the review

1. **Facts are checked deterministically.**
   - Extract the arXiv ids from the whole file, regardless of headings.
   - Check: the ids exist, are in the window, are not duplicated, have their text extracted, and number at least 5.
   - Fixes the parse cascade (C15): llama L2 timeseries/diffusion had 5 good papers and scored 0.1 because of `###` headings or literal `\n`.
2. **An LLM judge runs after the runs.**
   - Uses qwen 9B locally, blind to model and level, temperature 0, Pydantic structured output (numbers only).
   - Scores 1–5 on three questions: is it a valid review; are 2 randomly chosen summaries faithful to `texts/<id>.txt`; are the chosen papers relevant to the topic. The pass threshold is ≥4.
   - No manual validation.
   - This is the only measure that would catch the `'arXiv 2310'` review.
3. **Format is a separate metric.** The current strict checks are kept, but they do not decide success. The paper states that the code produces the format at L3.
4. **Two axes.**
   - Artifact quality: facts + judge, regardless of run status.
   - Agent reliability: did the agent finish on its own.
   - The old combined success stays as a third metric.
5. **Docstrings.** Replace the example ids (`2401.00001`) in the docstrings with a placeholder.
6. **Sampling.** Use explicit, identical `temperature`/`top_p`/`top_k` for all models and add them to the fingerprint.
   - Confirmed (C6): Qwen gets no truncation; gemma gets top_k 64 / top_p 0.95; llama gets top_p 0.9.
7. **Llama stays as it is**: one call per turn and hand-written JSON, listed as a limitation.
8. **Final run.** 3 repeats, 450 runs, on a new hash.

## C. Decisions to make: the review argues against earlier choices or raises new issues

1. **Reading depth (grilling Q13: "leave it as is").**
   - Every reviewer rates this as a methodology error, not a limitation.
   - At L1, qwen 9B never called `split_text_into_chunks`. `read_file` and `summarize_chunk` silently cut the text at 6000 characters (`utils/text.py` `truncate`, no marker).
   - L2/L3 summarize 6×3000 characters.
   - Consequences:
     - Cost differences between levels mostly reflect the amount of text processed, not granularity.
     - L3 summaries are 6 summaries concatenated (3–4k characters, while the prompt says "a few sentences").
   - Options:
     - (a) A truncation marker plus a report of characters read per paper.
     - (b) Equalize: `pipeline_max_chunks` matched to L1, or L1 forced through split.
     - (c) Leave it and describe it.
2. **The prompt describes the L1/L2 process, and the L3 tool description is incomplete.**
   - qwen 9B called the macro-tool again with the same arguments in 7 of 10 L3 runs ("let me check the content of the file"). This roughly doubles the L3 cost.
   - qwen 0.8B produced 9 loops.
   - Fix:
     - The task prompt describes the outcome, not the process.
     - The L3 docstring and result say plainly what was done ("5 papers read, review saved in the required structure, task complete").
3. **Parallel tool calls.**
   - Only llama has `parallel_tool_calls=false`. vLLM then drops the extra calls, and they are not counted.
   - qwen makes parallel calls: about 8 agent calls at L1, compared with about 29 for gemma.
   - Options:
     - (a) `false` for every model, the same policy everywhere.
     - (b) Leave it and compare costs only within a model.
4. **Model quantization.**
   - Current quantization: 9B AWQ-4bit (third party), gemma QAT W4A16 (Google), llama AWQ INT4, 4B fp8 online, 0.8B BF16.
   - Precision rises as model size falls, which understates the scale effect.
   - The 4B in BF16 would fit at gpu-mem 0.75.
   - Options:
     - (a) Serve the 4B in BF16.
     - (b) One scheme for everyone.
     - (c) Leave it and describe it.
5. **Tools differ between levels.**
   - L1 has 4 distractor tools, `read_file`, and append-only writing (5 L1 reviews with duplicated sections).
   - L2/L3 have no `read_file`.
   - L2 search returns an APA string with no abstract.
   - `methods.tex:62` says "the same capabilities". Options:
     - (a) Align: overwrite at L1, `read_file` everywhere or nowhere.
     - (b) Fix the methods text and describe the differences.
6. **H1b (task variants with an added requirement).**
   - There is no code and no scenarios for it, so the final run of 450 base tasks cannot test it.
   - Options:
     - (a) Implement the variants before the final run.
     - (b) Drop H1b from the paper, or move it to future work.
7. **Loop threshold and success.**
   - The screen used 5; the code now uses 8.
   - The threshold changes success for qwen-weak L3. With two axes (B4) this matters less, but the paper must state the value used.

## D. Fixes before the final run (protocol and infrastructure)

1. **Fingerprint.** Add `llm_max_tokens`, `llm_enable_thinking`, sampling and the server settings (checkpoint, quantization, parser, template hash, vLLM version). Today `runner.py` `config_fingerprint` omits them.
2. **Frozen caches.**
   - Pre-warm and freeze the arXiv and PDF caches for every scenario before the run; also cache 404s.
   - Unversioned PDF URLs return 404 (25× for qwen-weak). L3 silently substitutes another paper, while L1/L2 get a tool error.
3. **Retry `error` runs until every cell is complete.**
   - Two llama runs became `error` because arXiv failed during scoring, and were never retried, so llama has 9 scenarios.
   - Better: retry only the scoring, not the agent.
4. **Detect tool calls returned as text.** A reply without `tool_calls` whose content looks like a call. Today `invalid_tool_calls` is 0 in all 150 records.
5. **Count tool errors separately** for environment errors (404, 429) and agent errors.
6. **Warm-up request** to the model before the first run (Triton JIT at the start of gemma and llama).
7. **Do not overwrite `logs/server-<model>.log`.** The screen's qwen log was lost after a restart; keep one log per run invocation.
8. **`texts/` accumulates** across repeated L3 calls within a run (30 files), which weakens `papers_read`. Clear it, or count only the ids cited in the review.

## E. Analysis and statistics

1. **H1a "success no lower"** is currently judged with the superiority rule ("interval excludes 0"). Add a non-inferiority margin δ, for example 15 pp at R=3.
2. **Cost (tokens per success) has no interval** and is undefined when there are 0 successes (llama L1, qwen-weak L1/L2).
   - Use log tokens, geometric-mean ratios, and a cluster bootstrap over scenarios.
   - Declare up front which models H1a covers; models at the floor are reported as "not evaluable".
3. **Ceiling and floor cells.** "+0 pp [0, 0]" for qwen reads like precision. Use Wilson or Newcombe intervals, and a Firth/Bayesian GLMM if a model is needed.
4. **Bootstrap with 10 clusters undercovers on raw tokens** (simulation: 0.82). Use log tokens and a t-interval on 10 per-scenario differences (df=9) or a sign-flip permutation test.
5. **RQ2/RQ3 are interactions.** Test a difference of differences (level × model), not significance in each model separately. Fix the scale: risk difference for success, log for tokens.
6. **Repeat is a replicate, not a block** (each run has its own seed). Pair only by scenario.
7. **Tokens of stopped runs are censored** (loop, step limit, failed). Report them on completed runs and on all runs, plus a sensitivity analysis.
8. **Multiple comparisons.** One primary family (H1a) with Holm correction across models; everything else is exploratory.
9. **Cost metrics.**
   - Prefix caching is on, so record `cached_tokens` and report uncached tokens.
   - LLM seconds are summed per call, so parallel calls inflate them.
   - Compare time only within a model.
   - Tokens across model families depend on the tokenizer, so compare tokens only within a model.
10. **R=3 rather than 10.**
    - pass^k exists only for k ≤ 3.
    - The CI on a success difference is about ±15 pp.
    - Tokens have enough power: projected CIs are, for example, qwen L2/L1 0.83 [0.72, 0.95].

## F. Corrections to the paper text

- `methods.tex:62` "same capabilities": describe the differences (C5).
- "read in full" / "actually read" (`related_work.tex:8`, `data_environment.tex:7`, `methods.tex:127`, `Introduction.tex:17`, `data_environment.tex:65`): replace with "text extracted"; rename the check to `text_extracted`.
- `methods.tex:93` "quantized where needed" and the model's default top-p/top-k: update to the actual settings.
- The pilot table in `experiment_plan.tex` does not match the screen (for example Gemma L1 9/10 vs 8/10). Regenerate it or label it as an earlier version.
- R=3 instead of 10, and the status of H1b.
- Family claims are about specific checkpoints (Llama 3.1 from 2024 against Qwen3.5 and Gemma 4 from 2026).
- `search_papers` and the L3 tool say "recent", but they sort by relevance.
