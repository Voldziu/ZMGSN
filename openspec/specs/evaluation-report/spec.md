# evaluation-report Specification

## Purpose
Defines how the results files of evaluation runs are aggregated into the Markdown report and the CSV export: grouping by model and tool level, the quality, stability and cost metrics of the evaluation plan, and interval estimates for success rates and level differences.

## Requirements

### Requirement: The report combines results files by model and level
The `report` command SHALL accept one or more results files and summarise the scored runs of each model and level. Models SHALL appear in the order they first appear in the input, and levels in name order. Runs with an infrastructure error SHALL be excluded from every metric and listed separately.

#### Scenario: Several models
- **WHEN** a user runs `report results/presets-qwen.jsonl results/presets-llama.jsonl`
- **THEN** the report has an overview with one row per model and one column per level, and one row per model and level in each detailed table

#### Scenario: Infrastructure errors
- **WHEN** a model and level has 10 scored runs and 2 runs with an infrastructure error
- **THEN** its metrics use the 10 scored runs, and the 2 errors are counted and listed separately

### Requirement: Each run is counted once
The report SHALL reject input in which a scored run of the same model appears more than once, also when the two records come from different configurations. Its error SHALL name that run and, by file name, the files that hold it. A retried run, recorded once with an infrastructure error and once scored, SHALL NOT count as a duplicate.

#### Scenario: File passed twice
- **WHEN** a user passes the same results file twice
- **THEN** the report stops with an error that names a run appearing twice

#### Scenario: Two results files of one model
- **WHEN** a user passes `smoke-qwen.jsonl` and `presets-qwen.jsonl`, which both hold run `level1-rag-2024h1-01` of `qwen`
- **THEN** the report stops with an error that names the run and both files

#### Scenario: Retried run
- **WHEN** a results file holds an `error` record and a later scored record of the same run
- **THEN** the report counts the scored record once

### Requirement: Mixed configurations are flagged per model
When the scored runs of one model come from more than one configuration hash, the report SHALL warn that the levels of that model are not comparable. Runs of different models SHALL NOT trigger the warning.

#### Scenario: Two models
- **WHEN** the input holds `qwen` and `llama` runs, each with one configuration hash
- **THEN** the report shows no configuration warning

#### Scenario: One model with two configurations
- **WHEN** the `qwen` runs come from two configuration hashes
- **THEN** the report warns that the levels of `qwen` are not comparable

### Requirement: Success rate with a Wilson interval
For each model and level, the report SHALL give the number of successes, the success rate and its 95% Wilson score interval over the scored runs, the mean score and the pass rate of each check.

#### Scenario: Eight of ten
- **WHEN** a model and level has 8 successes in 10 scored runs
- **THEN** the report shows a success rate of 80% with the interval 49–94%

#### Scenario: All runs succeed
- **WHEN** all 10 scored runs succeed
- **THEN** the interval is 72–100%

### Requirement: Values without data are undefined
A rate or mean of a model and level without any scored run SHALL be undefined, and not zero. Repeatability SHALL be undefined when no scenario has two scored runs, and consistency when fewer than two scenarios have scored runs. The Markdown report SHALL show undefined values as `-`.

#### Scenario: Only infrastructure errors
- **WHEN** every run of a model and level ended with an infrastructure error
- **THEN** its success rate, interval, score, tokens, calls and times are undefined

#### Scenario: One repeat
- **WHEN** every scenario of a model and level has a single scored run
- **THEN** its repeatability is undefined

### Requirement: pass^k over the repeats of a scenario
For each model and level and each k from 1 to 10, the report SHALL give pass^k, the mean over scenarios of C(c,k)/C(n,k), where c is the number of successes in a scenario's n scored runs. The report SHALL leave pass^k undefined for any k above the smallest number of scored runs that a scenario has.

#### Scenario: Two scenarios
- **WHEN** one scenario has 10 successes in 10 runs and another has 5 in 10
- **THEN** pass^1 is 0.75, pass^2 is 0.61 and pass^10 is 0.50

#### Scenario: Too few repeats
- **WHEN** some scenario has only 3 scored runs
- **THEN** pass^k is given for k from 1 to 3 and is undefined above

### Requirement: Tokens per success
For each model and level, the report SHALL give the mean total, input and output tokens per scored run, including tool-internal calls, and the tokens per success: the mean total tokens divided by the success rate. Tokens per success SHALL be undefined when no run succeeded.

#### Scenario: Half of the runs succeed
- **WHEN** the runs of a model and level average 40,000 tokens and half of them succeed
- **THEN** the tokens per success are 80,000

#### Scenario: No success
- **WHEN** no run of a model and level succeeded
- **THEN** its tokens per success are undefined

### Requirement: Tool-call reliability and time
For each model and level, the report SHALL give the tool-error rate per tool call, the malformed-call rate per attempted tool call (tool calls plus malformed calls) and the mean number of malformed calls per run. It SHALL also give the means of the first-call input tokens, the agent and tool-internal model time, the remaining tool time and the wall time. A run whose record does not know a measurement SHALL be left out of that measurement's mean.

#### Scenario: Rates
- **WHEN** the runs of a model and level made 20 tool calls with 2 tool errors and 1 malformed call
- **THEN** the tool-error rate is 10% and the malformed-call rate is 1 in 21, about 4.8%

#### Scenario: Older records in the same file
- **GIVEN** a results file in which half of the records were written before model time was measured
- **WHEN** the report is rendered
- **THEN** the mean model time is the mean over the other half

### Requirement: CSV export for plots
`report --csv <file>` SHALL write a CSV file with a header row and one row per model and level. Each row SHALL contain every metric of the report, including the interval bounds, pass^1 to pass^10 and the pass rate of each check. Undefined values SHALL be empty cells, and the Markdown report SHALL still be printed.

#### Scenario: Export
- **WHEN** a user runs `report a.jsonl b.jsonl --csv summary.csv`
- **THEN** `summary.csv` has a header and one row per model and level, with columns for the success interval, pass^1 to pass^10 and tokens per success

#### Scenario: Undefined values
- **WHEN** no run of a model and level succeeded
- **THEN** its tokens-per-success cell is empty

#### Scenario: Level with only infrastructure errors
- **WHEN** every run of a model and level ended with an infrastructure error
- **THEN** its success-rate, score and mean cells are empty, not zero

### Requirement: The quality table counts runs by how they ended
For each model and level, the quality table SHALL show how many scored runs ended at the step limit, how many the loop guard stopped, and how many failed inside the agent, each in its own column.

#### Scenario: Loop runs
- **WHEN** qwen-weak at level 3 has 10 scored runs, of which 7 were stopped by the loop guard and 1 hit the step limit
- **THEN** its row shows 7 in the `loop` column and 1 in the `step limit` column

#### Scenario: Older results
- **WHEN** a results file written before the loop guard is reported
- **THEN** every row shows 0 in the `loop` column

### Requirement: Base tasks and variants are reported separately
The report SHALL summarise base tasks and variant tasks in separate sections and SHALL NOT pool them in any metric. Within the variant section it SHALL also break down the success rate by requirement. Level differences SHALL be computed within one task type only, pairing scenarios of that type.

#### Scenario: Results with both task types
- **WHEN** a results file holds base and variant runs of `qwen`
- **THEN** the report shows one quality and cost table for base tasks and one for variants, plus success by requirement for the variants

#### Scenario: Level 2 against level 3 on variants
- **WHEN** levels 2 and 3 of a model both have variant runs on ten variants
- **THEN** the report shows the variant `level3 − level2` difference computed over those ten variants only

#### Scenario: Only base runs
- **WHEN** a results file holds no variant runs
- **THEN** the report has no variant section

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
