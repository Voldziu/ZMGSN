# review-judging Specification

## Purpose
Defines the LLM judge, a separate step after the evaluation runs. It rates the structure, faithfulness and relevance of each review, and decides whether cited papers are surveys. It does so blind, deterministically, and with answers constrained to numbers.

## Requirements

### Requirement: The judge rates every scored run's review
A `judge` command SHALL take one or more results files and the run workspaces. It SHALL rate the review of every scored run with a served chat model (by default the `qwen` preset) on three 1–5 scales:
- **valid review:** one section and summary per paper;
- **faithfulness:** of two randomly chosen summaries, each against its paper's extracted text;
- **relevance:** of the chosen papers to the topic.

A run with no review file SHALL get no rating and SHALL count as not valid.

#### Scenario: Judging a results file
- **WHEN** a user runs `judge results/presets-gemma.jsonl` against a served `qwen`
- **THEN** every scored run of the file has a judgment with the three ratings

#### Scenario: Off-topic review
- **WHEN** a review on the topic "diffusion models image generation" summarizes papers on algebraic geometry and QCD
- **THEN** its relevance rating is below 4

### Requirement: The judge is blind to model and level
The judge's inputs SHALL be only the scenario topic, the review text, the requirement statement for variants, and the paper texts or metadata it rates. They SHALL NOT include the model, the level, the run id or any tool output.

#### Scenario: Prompt contents
- **WHEN** the judge prompt for a level-3 run of `llama` is built
- **THEN** it contains neither `llama` nor `level3` nor the run id

### Requirement: Judge answers are structured numbers only
Every judge call SHALL request output constrained to a fixed schema that holds only integers 1–5 or booleans, with no free text. A reply that does not satisfy the schema SHALL be retried once and otherwise recorded as missing.

#### Scenario: Valid answer
- **WHEN** the judge rates a review
- **THEN** the stored judgment holds integers for the ratings and no explanation text

#### Scenario: Malformed answer
- **WHEN** the model returns text that does not fit the schema twice
- **THEN** that rating is stored as missing and the run's artifact validity is undefined, not passed

### Requirement: Judging is deterministic and reproducible
The judge SHALL sample with temperature 0 and a fixed seed. The two summaries to rate SHALL be chosen by a seed derived from the run id, so that the same run always gets the same two. Each judgment SHALL record the judge model's identity and a hash of the judge prompts and settings.

#### Scenario: Re-judging
- **WHEN** the same results file is judged twice against the same server
- **THEN** the same summaries are chosen for each run, and the judgments carry the same judge hash

### Requirement: Survey decisions for `no_surveys` variants
For runs of a `no_surveys` variant, the judge SHALL decide for each cited paper whether it is a survey or review. It SHALL decide from the paper's arXiv title and abstract only, using a boolean answer.

#### Scenario: A survey is cited
- **WHEN** a `no_surveys` review cites "A Survey on Retrieval-Augmented Generation"
- **THEN** that paper is judged a survey, and the run's requirement is not met

### Requirement: Judgments are stored apart from run records
Judgments SHALL be written to a file next to each results file. Each judgment SHALL be keyed by run id and the run's configuration hash. Run records SHALL NOT be modified. Judging an already judged run with the same judge hash SHALL be skipped.

#### Scenario: Resumed judging
- **WHEN** `judge` is interrupted and started again with the same settings
- **THEN** it judges only the runs that have no judgment yet
