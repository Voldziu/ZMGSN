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
