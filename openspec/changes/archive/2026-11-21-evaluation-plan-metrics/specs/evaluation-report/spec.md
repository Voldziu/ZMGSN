# Spec Delta

## Purpose

Defines how the results files of evaluation runs are aggregated into the Markdown report and the CSV export: grouping by model and tool level, the quality, stability and cost metrics of the evaluation plan, and interval estimates for success rates and level differences.

## ADDED Requirements

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

### Requirement: Level differences with cluster-bootstrap intervals
Within each model and for each pair of levels, the report SHALL estimate the difference in success rate and in mean total tokens (later level minus earlier level) as the mean of the per-scenario differences over the scenarios both levels have. Every scenario SHALL weigh the same, whatever its number of runs. The report SHALL give a 95% percentile interval from 10,000 bootstrap resamples of those scenarios, reproducibly, and SHALL mark intervals that exclude zero.

#### Scenario: Two levels of one model
- **WHEN** levels 1 and 2 of `qwen` both have runs on 10 scenarios
- **THEN** the report shows `level2 − level1` with the number of scenarios and the differences in success rate and tokens, each with its interval, and marks every interval that excludes zero

#### Scenario: Reproducible
- **WHEN** the report is rendered twice from the same files
- **THEN** the intervals are identical

#### Scenario: Scenario missing at one level
- **WHEN** a scenario has runs at level 1 but none at level 2
- **THEN** that scenario is left out of the `level2 − level1` comparison

#### Scenario: Unequal numbers of runs
- **GIVEN** levels 1 and 2 have the same success rate in every scenario, but level 2 has only 2 scored runs in one of them
- **WHEN** they are compared
- **THEN** the success difference is 0 with the interval 0 to 0

#### Scenario: No difference
- **WHEN** every run of both levels succeeded
- **THEN** the success difference is 0 with the interval 0 to 0, and it is not marked

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
