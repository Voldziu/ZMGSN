# Spec Delta

## ADDED Requirements

### Requirement: One summary per paper at levels 2 and 3
When a level-2 or level-3 tool summarizes a paper, it SHALL first summarize the paper's leading chunks and then make one further model call that combines the chunk summaries into a single summary of a few sentences. The tool SHALL return and store only that combined summary. Chunk summaries SHALL NOT be concatenated into the review.

#### Scenario: Level-3 section length
- **WHEN** the level-3 tool writes a review of five papers, each summarized from six chunks
- **THEN** each section holds one summary of a few sentences, not six joined chunk summaries

#### Scenario: Level-2 tool result
- **WHEN** the agent calls `get_and_summarize_paper`
- **THEN** the returned summary is the combined summary, and the number of chunks summarized is reported alongside it

### Requirement: The level-3 tool states that the task is done
The level-3 tool's description SHALL state what the tool does. It reads the full text of the requested number of papers from the date window, summarizes each one, and saves `reviews/review.md` in the structure the task requires. After a successful call, its result SHALL state plainly that the review is complete and saved, and SHALL give the number of papers it read in full. Level 3 SHALL offer no other tool.

#### Scenario: Successful call
- **WHEN** the level-3 tool includes 5 read papers in the review
- **THEN** its result says that the review with 5 papers read in full is saved to `reviews/review.md` and the task is complete

#### Scenario: Shortfall
- **WHEN** fewer than the requested papers could be read
- **THEN** the result says how many were included and that the review is incomplete

### Requirement: A repeated level-3 call starts from a clean workspace
Each call of the level-3 tool SHALL first remove the extracted texts, chunks and summaries left by an earlier call in the same run. The run workspace then reflects only the papers of the last call.

#### Scenario: Tool called twice
- **WHEN** the agent calls the level-3 tool twice with different queries
- **THEN** after the second call, `texts/` holds only the texts of papers read by the second call
