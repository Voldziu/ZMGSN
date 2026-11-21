# Spec Delta

## ADDED Requirements

### Requirement: The report shows both outcome axes and format compliance
For each model, level and task type, the report SHALL give rates with Wilson intervals for:
- artifact validity;
- reliability;
- success;
- format success.

It SHALL also give the mean of each judge rating and the pass rate of each fact check and format check. When judgments are missing for some runs, the report SHALL say for how many runs, and SHALL compute artifact validity and success over the judged runs only.

#### Scenario: Judged results
- **WHEN** the report reads a results file and its judgment file
- **THEN** it shows artifact validity, reliability, success and format success, each with an interval

#### Scenario: No judgments yet
- **WHEN** no judgment file exists for a results file
- **THEN** the report shows reliability, format success and the fact-check rates, and marks artifact validity and success as not available

### Requirement: Level differences with scenario-paired intervals
Within each model and task type, and for each pair of levels, the report SHALL estimate the difference in each outcome rate as the mean of the per-scenario differences over the scenarios that both levels have, with every scenario weighing the same. It SHALL give a 95% t-interval with (scenarios − 1) degrees of freedom and an exact sign-flip permutation p-value.

#### Scenario: Two levels of one model
- **WHEN** levels 1 and 2 of `gemma` both have base runs on 10 scenarios
- **THEN** the report shows `level2 − level1` for each outcome, with the number of scenarios, a t-interval and a permutation p-value

#### Scenario: Scenario missing at one level
- **WHEN** a scenario has runs at level 1 but none at level 2
- **THEN** that scenario is left out of the `level2 − level1` comparison

### Requirement: Comparisons at the ceiling or floor are labelled
When both levels of a comparison are at 100% in every scenario, or both at 0%, the report SHALL state "both at ceiling" or "both at floor" instead of an interval.

#### Scenario: Ceiling
- **WHEN** every run of levels 1 and 2 of `qwen` is a success
- **THEN** the success comparison reads "both at ceiling"

### Requirement: Tokens are compared as ratios on the log scale
Level differences in tokens SHALL be computed on the logarithm of the total tokens per run. They SHALL be reported as the geometric-mean ratio with the same paired interval. Ratios SHALL be given for all scored runs and for completed runs only.

#### Scenario: Token ratio
- **WHEN** level 2 of a model uses about 40% of the tokens of level 1 in every scenario
- **THEN** the report shows a ratio of about 0.40 with an interval, for all runs and for completed runs

### Requirement: Tokens per success with an interval
The report SHALL give the ratio of tokens per success between two levels, with a 95% BCa interval from a bootstrap that resamples scenarios jointly for both levels. The ratio SHALL be "not evaluable" when either level has no success.

#### Scenario: No success at level 1
- **WHEN** no level-1 run of `llama` is a success
- **THEN** the `level2 / level1` tokens-per-success ratio of `llama` reads "not evaluable"

### Requirement: H1a is tested as non-inferiority with Holm correction
For each model, the report SHALL state whether level 2 is non-inferior to level 1 in success on base tasks. Non-inferiority holds when the lower bound of the `level2 − level1` interval is above −15 percentage points. The report SHALL also state whether the tokens-per-success ratio's upper bound is below 1. It SHALL support H1a for a model only when both conditions hold after Holm correction across the models.

#### Scenario: Non-inferior and cheaper
- **WHEN** a model's `level2 − level1` success interval is −8 to +12 percentage points, and its tokens-per-success ratio is 0.45 with an interval of 0.35 to 0.60, both surviving Holm correction
- **THEN** the report marks H1a as supported for that model

### Requirement: Level-by-model interactions
The report SHALL give, for each pair of models and each pair of levels, the difference between the models' level differences. It SHALL use a 95% interval from a bootstrap that resamples scenarios jointly across both models. Success SHALL be compared on the risk-difference scale and tokens on the log scale.

#### Scenario: Scale series
- **WHEN** `qwen`, `qwen-mid` and `qwen-weak` all have runs at levels 1 and 3
- **THEN** the report shows, for example, `(qwen-weak: level3 − level1) − (qwen: level3 − level1)` with an interval

### Requirement: Excluded runs are counted
For each model, level and task type, the report SHALL give the number of runs excluded as infrastructure errors. It SHALL also give a sensitivity line with the success rate when those runs are counted as failures.

#### Scenario: Two arXiv errors
- **WHEN** two llama level-1 runs ended as `error`
- **THEN** the report shows 2 excluded runs for that cell and the success rate with them counted as failures

## REMOVED Requirements

### Requirement: Level differences with cluster-bootstrap intervals
**Reason**: With 10 skewed clusters, the percentile bootstrap undercovers (simulated coverage 0.82 on tokens). It also prints a misleading "0 [0, 0]" at the ceiling.
**Migration**: The new requirements "Level differences with scenario-paired intervals" and "Tokens are compared as ratios on the log scale" replace it.
