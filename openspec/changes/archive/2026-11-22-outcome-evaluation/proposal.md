# Proposal

## Why

The methodology review of 2026-11-21 (`docs/methodology-review-2026-11-21.md`) found that the current success metric does not measure review quality that can be credited to the model and tool system.

- **Level 3 passes by construction.** At level 3 the code writes the headings, citations, bibliography and texts, so all 50 level-3 screen runs passed every check. One of them was an off-topic review (query `'arXiv 2310'`) that scored 1.0.
- **Format slips fail sound reviews.** The ten checks parse sections from `## ` headings, so one formatting slip fails up to nine checks. Llama reviews with five real, read papers scored 0.1 because they used `###` headings.
- **Content is never checked.** Topical relevance and faithfulness of the summaries are not measured.
- **Two outcomes are mixed.** "Agent stopped cleanly" and "review is valid" are folded into one success flag. Qwen 0.8B wrote a valid level-3 review in 9 of 10 runs and still scored 10%.
- **The statistical plan is weak.** H1a's "no lower success" is tested as superiority, without a margin. Tokens per success has no interval. Ceiling cells print a misleading "0 [0, 0]". The cluster bootstrap under-covers with 10 skewed clusters, and RQ2/RQ3 compare significance instead of testing an interaction.

The group decided:
- check facts deterministically;
- judge structure and content with a blind LLM judge (Qwen3.5-9B, local, structured output, no justification);
- keep the strict format checks as a separate metric;
- report two axes, artifact quality and agent reliability;
- fix the analysis plan.

The scoring changes work on saved run files only, so they can be built while the final run is in progress.

## What Changes

- **Fact checks.** Deterministic checks read every arXiv id anywhere in the review file, regardless of headings or formatting, and check that:
  - at least N distinct papers are cited;
  - every id exists;
  - every id is in the date window;
  - every cited paper's text was extracted in the run (`text_extracted`).

  They are computed at scoring time and stored next to the format checks.
- **Requirement check for variants** (`requirement_met`):
  - `table`: deterministic, a Markdown table between the main heading and the first paper section, with a row for every cited paper;
  - `polish`: deterministic, language detection on the summary text, excluding titles and references;
  - `no_surveys`: decided by the judge from each cited paper's arXiv title and abstract.
- **LLM judge, a separate step after the runs.**
  - A `judge` command sends each scored run's review to a served model (Qwen3.5-9B by default) without revealing the model or level.
  - It rates three things on 1–5:
    - whether the file is a valid literature review with one section and summary per paper;
    - the faithfulness of two randomly chosen summaries to the paper's extracted text;
    - the relevance of the chosen papers to the topic.
  - For `no_surveys` variants it also decides survey or not for each cited paper.
  - The answers are constrained to Pydantic models that hold numbers or booleans only, with no reasons. Temperature is 0 and the seed is fixed.
  - Judgments go to a separate file next to the results, keyed by run id and configuration hash, so the run records stay raw.
- **Three outcomes per run:**
  - **artifact valid**: all fact checks pass, every judge rating is ≥ 4, and, for variants, the requirement is met. Run status does not matter.
  - **reliable**: the agent ended the run itself (`completed`).
  - **success**: both of the above.

  The current rule (completed and all ten format checks) stays as **format success**, and the ten checks are reported as format compliance.
- **`rescore` command.** It recomputes the deterministic checks (format, facts, table and Polish requirements) from saved run workspaces, so earlier results files, including the screen, can be scored under the new rules.
- **Analysis and report (BREAKING for report output).**
  - **Success rates:** Wilson intervals. A level difference is the mean of per-scenario differences with a t-interval (df = scenarios − 1) and an exact sign-flip permutation p-value; this replaces the percentile bootstrap. Two cells at the same bound (both 0% or both 100%) are reported as "at ceiling" or "at floor", not as "0 [0, 0]".
  - **Tokens:** analysed on the log scale and reported as geometric-mean ratios with the same paired interval. Shown for all runs and for completed runs only.
  - **Tokens per success:** a ratio between levels with a scenario-cluster bootstrap BCa interval. Reported as "not evaluable" when either level has no success.
  - **H1a non-inferiority:** level 2 is non-inferior to level 1 in success when the lower bound of the `level2 − level1` interval exceeds −15 percentage points. Holm correction is applied across models within the H1a family.
  - **RQ2/RQ3:** level × model interaction as a difference of differences, resampling scenarios jointly across models.
  - **Excluded runs:** counts per model, level and task type, plus a sensitivity line that counts them as failures.
- **Paper.**
  - `methods.tex` (evaluation section) defines the fact checks, the judge, the three outcomes and the format metric.
  - `experiment_plan.tex` states the primary outcome, the margin, the tests, the Holm family, and that everything else is exploratory.
  - `progress-6-7.html` shows the new metrics.

## Capabilities

### New Capabilities
- `review-judging`: the LLM judge — its inputs, blinding, ratings, structured output, determinism and judgment file.

### Modified Capabilities
- `review-scoring`:
  - fact checks over the whole file;
  - the requirement check for variants;
  - the three outcomes and format success;
  - the `rescore` command.
- `evaluation-report`:
  - two-axis outcomes;
  - format compliance;
  - paired t-intervals and permutation p-values instead of the percentile bootstrap;
  - log-token ratios;
  - tokens-per-success intervals;
  - H1a non-inferiority with Holm;
  - interaction contrasts;
  - excluded-run counts;
  - ceiling and floor reporting.

## Impact

- **Code:**
  - `evaluation/checks.py` (fact and requirement checks), `evaluation/models.py` (outcome groups, `artifact_valid`, `reliable`, `format_success`);
  - new `evaluation/judge.py` and `evaluation/rescore.py`;
  - `evaluation/stats.py` (t-interval, sign-flip permutation, BCa, Holm), `evaluation/report.py`;
  - `main.py` (`judge`, `rescore`);
  - new prompts `prompts/judge_{review,summary,survey}/v1.md`.
- **Dependencies:** `langdetect` (Polish detection).
- **Results:** the run-time scoring version changes (CHECKS_VERSION 3). Judgment files are new artifacts. Older results can be rescored.
- **Process:** `judge` needs the `qwen` preset served after the final run, and the run workspaces (`workspace/preset-runs/<model>/`) must be kept.
- **Depends on:** `task-variants` (requirement field on scenarios and records).
- **Docs:** `README.md`, `docs/overleaf/methods.tex`, `docs/overleaf/experiment_plan.tex`, `docs/presentations/src/progress-6-7.html`.
