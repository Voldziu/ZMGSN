## 1. Outcome model

- [x] 1.1 Add `facts` and `requirement` to `OutcomeScore`. Add `format_success` and `reliable` to `RunRecord`, plus an `artifact_valid(judgment)` helper. Keep `success` equal to `format_success`. Old records load.
- [x] 1.2 Raise `CHECKS_VERSION` to 3 and update the fingerprint tests.

## 2. Deterministic checks

- [x] 2.1 Add `_cited_ids` (abs/pdf URLs and `arXiv:` ids, versions stripped) and the four fact checks.
  - Use the screen cases as fixtures: llama `###` headings, a one-line file with literal `\n`, ids only in the bibliography, invented ids.
- [x] 2.2 Add the `table` requirement check. Tests: a table in the right place passes; a missing table fails; a table missing a paper fails; a table after the first paper section fails.
- [x] 2.3 Add the `polish` requirement check with `langdetect` (seeded), and add `langdetect` to the dependencies. Tests: Polish summaries pass; English fails; titles and citations are ignored.
- [x] 2.4 Wire the facts and the requirement into `evaluate_review` (`no_surveys` stays undefined).

## 3. Judge

- [x] 3.1 Add Pydantic schemas `ReviewRating`, `SummaryRating` and `SurveyDecision`, and the prompts `judge_review`, `judge_summary` and `judge_survey` (v1).
- [x] 3.2 Add `evaluation/judge.py`:
  - inputs: topic, requirement, review text, sampled sections plus source texts, metadata for surveys;
  - structured calls with one retry;
  - seeded section choice;
  - `judge_hash`;
  - `<results>.judge.jsonl` writing with skip-on-resume.

  Tests use a fake structured LLM: blinding (no model, level or run id in the prompts), determinism, the missing-rating path, resume.
- [x] 3.3 Add the CLI `tool-eval judge` and the Makefile target `make judge PRESETS=... JUDGE=qwen`.

## 4. Rescore

- [x] 4.1 Add `evaluation/rescore.py` and the CLI `tool-eval rescore`. Tests: the new file has facts added, the original is unchanged, and a missing workspace is listed.

## 5. Statistics and report

- [x] 5.1 Add `paired_t_interval`, `sign_flip_p` (exact and Monte Carlo), `bca_interval` and `holm` to `stats.py`, with tests against known values.
- [x] 5.2 Rework `compare_levels`:
  - per task type;
  - all four outcomes;
  - t-interval and p-value;
  - ceiling and floor labels;
  - log-token ratios (all and completed);
  - tokens-per-success ratio with BCa.

  Remove the percentile-bootstrap difference and update `tests/test_report.py`.
- [x] 5.3 Add the H1a box (non-inferiority with δ = 15 pp, the cost criterion, Holm across models) and the H1b comparison (variants, `level2 − level3`).
- [x] 5.4 Add the interaction contrasts for model pairs and level pairs, resampling scenarios jointly.
- [x] 5.5 Add excluded-run counts and the sensitivity line. Show judge ratings and fact pass rates. Have the report find `<results>.judge.jsonl` automatically.
- [x] 5.6 CSV: add the new columns and one row per model, level and task type.

## 6. Docs, paper and presentation

- [x] 6.1 `README.md`: the three outcomes, the fact checks, `judge`, `rescore` and the report sections.
- [x] 6.2 `docs/overleaf/methods.tex`, evaluation section:
  - the fact checks;
  - the requirement checks;
  - the judge (model, blinding, scales, threshold, structured output, determinism, self-preference limitation);
  - the three outcomes and format compliance.
- [x] 6.3 `docs/overleaf/experiment_plan.tex`:
  - primary outcome and family;
  - H1a non-inferiority (δ = 15 pp) with the cost criterion and Holm;
  - H1b on variants;
  - RQ2/RQ3 as interactions;
  - paired t-interval and sign-flip test;
  - log tokens;
  - tokens per success with BCa;
  - excluded runs and sensitivity;
  - everything else exploratory.
- [x] 6.4 `docs/presentations/src/progress-6-7.html`: a slide on the new metrics (facts, judge, two axes) and the analysis plan. Regenerate the PDF.

## 7. Verification

- [x] 7.1 `make test`, ruff and mypy clean.
- [x] 7.2 Rescore the screen results in `results/old/screen-1121/` on the RTX and judge them with a served `qwen`. Check two things:
  - the off-topic `qwen-weak level3-diffusion` run is not artifact valid;
  - llama's `###`-heading reviews pass the fact checks.

  Record the outcome in this file.

Notes on 7.2 (2026-11-22). At the user's request this ran locally on the Mac with qwen-mid as the judge, not on the RTX with qwen. The RTX was freed after the variant smoke run.
- **Structured output:** works on vllm-metal (`json_schema` guided decoding). One judgment takes about 4 s.
- **Screen edge cases** (copied from `results/old/screen-1121` on the RTX):
  - The off-topic qwen-weak `level3-diffusion` review ("arXiv 2310"), format score 1.0, got relevance 1, so it is not artifact valid.
  - llama `level2-timeseries` (`###` headings, format score 0.1) passes the fact checks and gets 5/5/5, so it is artifact valid. Under the old rule this was a false negative.
  - llama `level1-rlhf` (placeholders and invented ids) fails the fact checks and gets 1/1/1.
  - llama `level2-diffusion` is a one-line file with literal `\n` escapes. Its fact checks pass, but no paper section can be rated, so it now counts as not valid. `Judgment.passes` was changed for this case.
- **Variant smoke (RTX runs, judged locally):**

  | level | success | format success |
  |---|---|---|
  | L1 | 2/3 | 2/3 |
  | L2 | 3/3 | 3/3 |
  | L3 | 1/3 | 3/3 |

  L3 fails `table` and `polish` and passes `no_surveys`. No cited paper was judged a survey.
- **Base smoke (Mac):** 9 of 9 judged runs are artifact valid. The judge (qwen-mid) is lenient: nearly every rating is 5.
