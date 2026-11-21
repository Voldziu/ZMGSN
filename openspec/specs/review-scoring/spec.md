# review-scoring Specification

## Purpose
Defines what an evaluation run must achieve to count as successful. That is a review that meets the task's format and is built from papers the agent actually read, so that scores reflect the work done and not only the shape of the file.

## Requirements

### Requirement: The task requires summaries from full text
The evaluation task SHALL instruct the agent to obtain each cited paper's full text with the tools and to base its summary on that text. The task SHALL say that a summary based only on search results or abstracts does not count.

#### Scenario: Task text states the requirement
- **WHEN** the evaluation task prompt is rendered for any scenario
- **THEN** it states that each paper must be read in full with the tools before it is summarized

### Requirement: Cited papers must have been read
Every evaluation run SHALL be scored with a `papers_read` criterion. The criterion SHALL pass only if, for every arXiv paper cited in the review, the run's own workspace contains that paper's extracted full text. It SHALL fail when any cited paper was not read in that run, and its detail SHALL name the unread papers.

#### Scenario: Review written from search results only
- **GIVEN** the agent searched arXiv and wrote a five-section review without downloading any paper
- **WHEN** the run is scored
- **THEN** `papers_read` fails, naming all five papers, and the run is not successful

#### Scenario: One paper could not be read
- **GIVEN** the agent read four papers and cited a fifth whose PDF failed to download
- **WHEN** the run is scored
- **THEN** `papers_read` fails and names the fifth paper

#### Scenario: All cited papers read
- **WHEN** every cited paper's full text was extracted in the run's workspace
- **THEN** `papers_read` passes

#### Scenario: Files from other runs do not count
- **WHEN** a cited paper's PDF exists only in the shared PDF cache, and its text was not extracted in this run
- **THEN** `papers_read` fails for that paper

### Requirement: Only completed runs succeed
A run SHALL count as a success only when its status is `completed` and every check passes. A run that hit the step limit, was stopped by the loop guard or failed SHALL NOT count as a success, even if every check passes, but its score SHALL still count as partial credit. The run record SHALL state whether the run succeeded, and the rule SHALL also apply to records written before it existed.

#### Scenario: Completed run passing every check
- **WHEN** a run ends with status `completed` and every check passes
- **THEN** it counts as a success with score 1.0

#### Scenario: Step limit after a complete review
- **GIVEN** the agent wrote a review that passes every check and then kept calling tools until the step limit
- **WHEN** the run is scored
- **THEN** it does not count as a success, and its score is 1.0

#### Scenario: Loop after a complete review
- **GIVEN** the agent wrote a review that passes every check and then requested the same review call until the loop guard stopped it
- **WHEN** the run is scored
- **THEN** its status is `loop`, it does not count as a success, and its score is 1.0

#### Scenario: Failed run keeps partial credit
- **WHEN** a run fails because the context window overflowed, and 6 of its 10 checks pass
- **THEN** it does not count as a success, and its score is 0.6

#### Scenario: Older results
- **GIVEN** a results file written before this rule, holding a `step_limit` run whose checks all passed
- **WHEN** the file is reported
- **THEN** that run does not count as a success
