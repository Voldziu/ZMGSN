# Spec Delta

## ADDED Requirements

### Requirement: Prompts state the outcome, not the tool steps
The system and task prompts SHALL be the same at every level. They SHALL describe what the finished review must contain: the number of papers, the date window, summaries based on each paper's full text, and the file structure. They SHALL NOT prescribe tool operations such as downloading, extracting or splitting. The tool descriptions alone SHALL tell the agent how its level reaches the outcome.

#### Scenario: Task prompt has no process steps
- **WHEN** the task prompt is rendered for any scenario
- **THEN** it does not instruct the agent to download, extract, split or summarize text, and it states that every summary must reflect the paper's full text

#### Scenario: Same prompt at every level
- **WHEN** the same scenario is run at levels 1, 2 and 3
- **THEN** the agent receives an identical system prompt and task prompt

### Requirement: Tool descriptions use placeholder identifiers
A tool description SHALL NOT show an identifier that looks like a real arXiv paper as an example. Where an example helps, it SHALL use an obvious placeholder that states where real identifiers come from.

#### Scenario: Download tool description
- **WHEN** the description of any tool that takes an arXiv identifier is rendered
- **THEN** it contains no string of the form `NNNN.NNNNN` and tells the agent to use an identifier returned by a search

### Requirement: Level 1 writes files by replacing their content
Level 1 SHALL offer a `write_file` tool that replaces the whole content of a Markdown or text file in the workspace, creating the file if needed. Level 1 SHALL NOT offer an append-only writing tool. The level-1 distractor tools (Wikipedia search, keyword extraction, translation and Markdown tables) SHALL remain.

#### Scenario: Correcting the review at level 1
- **WHEN** the agent writes `reviews/review.md` with `write_file` and later writes it again with a corrected review
- **THEN** the file contains only the corrected review

#### Scenario: Level-1 tool set
- **WHEN** the level-1 tools are listed
- **THEN** they include `write_file` and the four distractor tools, and no append tool
