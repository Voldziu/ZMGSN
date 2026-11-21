# Design

## Context

Motivation and decisions: proposal.md, `docs/methodology-review-2026-11-21.md`, and the grilling decisions of 2026-11-21. Current state:

- `evaluation/checks.py`:
  - `evaluate_review` parses sections from `## ` lines (`_parse`) and takes ids only from parsed sections.
  - It runs ten `CRITERIA`, and `CHECKS_VERSION = 2`.
  - `ArxivLookup` answers id metadata from the shared arXiv cache.
- `evaluation/models.py`:
  - `OutcomeScore` holds `criteria`, `score` and `success`.
  - `RunRecord.success` = `status == "completed" and outcome.success`.
- `evaluation/report.py` and `stats.py`:
  - Success rates use Wilson intervals.
  - Level differences use `cluster_bootstrap` with percentile intervals on success and raw mean tokens.
  - Tokens per success = mean tokens / success rate, with no interval.
- After `protocol-fixes` and `task-variants`:
  - records carry the task type and requirement;
  - run workspaces stay under `workspace/preset-runs/<model>/<run_id>/`, with `reviews/review.md` and `texts/<id>.txt`;
  - every request samples with explicit settings.

## Goals / Non-Goals

**Goals**
- Score content and provenance independently of formatting.
- Measure relevance and faithfulness with a blind, deterministic judge that returns numbers only.
- Keep run status apart from artifact quality.
- Report analyses that match the hypotheses: non-inferiority, ratios, interactions, ceiling and floor.

**Non-Goals**
- Human validation of the judge. The user rejected it; the paper states this as a limitation.
- A judge from outside the evaluated pool. The user accepted the self-preference risk of using Qwen3.5-9B; the paper discloses it.
- Mixed-effects models (GLMM/LMM). The paired scenario-level analysis is the primary method. A mixed model may be added later as exploratory work.

## Decisions

### Outcome structure
- `OutcomeScore` gains `facts: list[CriterionResult]` and `requirement: CriterionResult | None`. The ten existing criteria stay in `criteria`, which now means "format". `score` stays the format score.
- `RunRecord` gains computed properties:
  - `format_success`: the old rule.
  - `reliable`: `status == "completed"`.
  - `artifact_valid(judgment) -> bool | None`, which combines facts, the deterministic requirement and the judgment.
- `success` stays as a stored field equal to `format_success`, so old reports and records keep loading. The report uses the new properties.
- `CHECKS_VERSION` goes to 3.

### Fact checks
- `_cited_ids(text)` collects every match of:
  - `arxiv.org/(abs|pdf)/<id>(vN)?`;
  - `arXiv:<id>`.

  It keeps the first-seen order and strips versions. The id pattern is the existing `_ID`.
- The four checks run on that set:
  - `papers_cited` passes when the set has at least N ids;
  - `cited_ids_exist` uses the lookup;
  - `cited_in_window` uses `_in_window`;
  - `text_extracted` checks for `texts/<stem>.txt` in the run workspace.
- Duplicates are not a fact failure, because a paper is normally cited in its section and again in the bibliography. The judge's "valid review" rating covers repeated sections.

### Deterministic requirement checks
- **`table`:**
  - In the text between the first `# ` line and the first paper section, find a Markdown table: a header row, a separator row `|---|`, and data rows.
  - The first paper section is the first heading of any level that is followed by a cited id before the next heading.
  - Pass when every cited id appears in a row, either as the id or as its normalised arXiv title.
- **`polish`:**
  - For each paper block (text between consecutive headings that contains a cited id), remove lines containing URLs, the citation line and the heading.
  - Run `langdetect` (seeded with `DetectorFactory.seed = 0`) on the rest.
  - Pass when every block with at least 40 characters of remaining text is detected as `pl`.
- **`no_surveys`:** left undefined until the judgment exists.

### Judge
- **Module and command.** `evaluation/judge.py` and the `tool-eval judge <results...> --runs-dir <dir> [--judge-model qwen] [--base-url …]`. The judge uses its own `ChatOpenAI` with temperature 0, seed 0, thinking off, `top_p` 1 and `max_completion_tokens` 64.
- **Structured output.** `llm.with_structured_output(Model, method="json_schema")`. vLLM enforces the schema through guided decoding. The models are:
  - `ReviewRating(valid_review: conint(1,5), relevance: conint(1,5))`;
  - `SummaryRating(faithfulness: conint(1,5))`;
  - `SurveyDecision(is_survey: bool)`.

  If parsing fails, the call is retried once; if it fails again, the rating is stored as missing.
- **Calls per run:**
  - one review call with the topic, the requirement statement (variants), and the review text up to 20,000 characters, with the cut marker;
  - two summary calls, each with one section's summary and its paper's extracted text up to 12,000 characters;
  - for `no_surveys` variants, one survey call per cited paper with the arXiv title and abstract from the lookup.

  Sections are chosen with `random.Random(sha256(run_id))`. A review with fewer than two paper sections gets ratings for those it has.
- **Prompts.** `prompts/judge_review/v1.md`, `judge_summary/v1.md` and `judge_survey/v1.md`. Each gives a short rubric with anchors for 1, 3 and 5 and asks for the schema only. Their hashes, the judge model identity (as in `protocol-fixes`) and the settings form the `judge_hash`.
- **Output file.** `<results>.judge.jsonl`, one `Judgment(run_id, config_hash, judge_hash, ratings…, surveys: dict[id, bool] | None)` per run. A run that already has a judgment with the same `judge_hash` is skipped.
- **Blinding.** The judge prompt never includes the model, the level or the run id. One limitation remains: level-3 reviews have a recognisable fixed style ("Source: arXiv. Submitted between…"). The paper states this.
- *Alternative:* a single call rating everything. Rejected, because faithfulness needs the paper text, and five full papers do not fit a 32k context.

### Artifact validity
A run is artifact valid when all of the following hold:
- all facts pass;
- `valid_review ≥ 4`;
- `relevance ≥ 4`;
- the minimum of the faithfulness ratings is ≥ 4;
- for a variant, the requirement passes (for `no_surveys`: no cited paper is a survey).

Any missing piece makes it undefined. Undefined runs are excluded from the rate and counted in the report.

### Rescore
`evaluation/rescore.py` and `tool-eval rescore <results> --runs-dir <dir> [--out <file>]` rebuild each scenario from the scenarios file and rerun `evaluate_review` on the saved workspace with the cached lookup. They write `<results>.rescored.jsonl`.

### Statistics (`stats.py`)
New functions:
- `paired_t_interval(diffs) -> (mean, low, high)`;
- `sign_flip_p(diffs)`, exact for n ≤ 20 by enumerating 2^n sign flips, Monte Carlo with 100,000 draws above that;
- `bca_interval(statistic, clusters, n_boot=10000, seed)` for ratios of tokens per success;
- `holm(p_values)`.

How the analyses use them:
- **Level differences:** rates per scenario per level, then differences, then `paired_t_interval` and `sign_flip_p`.
- **Tokens:** the mean of log tokens per scenario per level, then differences, then exp of the interval, giving a ratio.
- **Tokens per success:** the ratio of sum(tokens)/sum(successes) between levels, with scenarios resampled jointly; "not evaluable" if either level has 0 successes.
- **H1a:** success non-inferiority on base tasks with margin δ = 15 percentage points. A one-sided p-value comes from the sign-flip test of `diff + δ`. The cost criterion takes the p-value from the sign-flip test on log TPS per scenario where defined, otherwise from the bootstrap. The union–intersection p is the max of the two, and Holm is applied across models.
- **Interactions:** for models A and B and levels i and j, the bootstrap resamples scenarios jointly and computes `(A_j − A_i) − (B_j − B_i)`. Success uses the risk difference; tokens use the log ratio.

The 15-pp margin is set before the run. With R=3 the success interval half-width is about 15 pp, so smaller margins could not be demonstrated.

### Report layout
For each task type:
- outcome table (artifact valid, reliable, success, format success);
- judge ratings;
- fact and format pass rates;
- cost (all runs and completed only);
- level comparisons;
- H1a box (base tasks) and H1b comparison (variants, `level2 − level3`);
- interactions;
- excluded runs.

The CSV gains every new column.

## Risks / Trade-offs

- **[Self-preference of the Qwen judge.]** Accepted by the user and disclosed. Blinding reduces it but cannot remove it.
- **[Judge leniency or harshness changes the levels of artifact validity.]** Comparisons between levels use the same judge, so differences are less affected than absolute rates. The ratings are also reported as means, not only through the ≥ 4 threshold.
- **[`langdetect` misclassifies short or mixed text.]** Blocks shorter than 40 characters are skipped, and URLs and citations are removed. Unit tests cover English and Polish samples.
- **[Judging costs GPU time.]** About 3–8 calls per run and about 5,000 calls for 900 runs, all short outputs. At an estimated 2–3 hours it runs after the final run.
- **[Run workspaces must survive.]** Judging and rescoring need `workspace/preset-runs/`, so it must not be cleaned until judging is done.

## Migration Plan

1. Implement with unit tests. Rescore the screen (`results/old/screen-1121/`) as a dry run of the new checks and report.
2. On the RTX after the final run, serve `qwen`, then run `judge` on all five results files, then `report`.
3. Put the report's numbers into the paper (`results-writeup`, a later change).

Rollback: the new outcomes are additive, and `format_success` keeps the old metric.

## Open Questions

None. The margin (15 pp), the threshold (≥ 4) and the judge model (Qwen3.5-9B) were settled in the grilling session.
