# agent-tool-calling Specification

## Purpose
Defines how the evaluation agent executes tool calls: how many model calls a run may take, and how tool arguments from the model are prepared before the tool runs.

## Requirements

### Requirement: Default step limit
A run SHALL allow at most 100 model calls by default. `AGENT_MAX_STEPS` SHALL override the limit. A run that reaches the limit without finishing SHALL be recorded with status `step_limit`.

#### Scenario: Default limit
- **WHEN** `AGENT_MAX_STEPS` is not set
- **THEN** a run may take up to 100 model calls before it is stopped

#### Scenario: One-tool-per-turn model finishes
- **GIVEN** a model that calls one tool per reply and needs about 33 calls for the level-1 scenario
- **WHEN** it runs with the default limit
- **THEN** the run can end with status `completed`

#### Scenario: Runaway run is stopped
- **WHEN** a model keeps calling tools without finishing
- **THEN** the run stops after 100 model calls and is recorded as `step_limit`

### Requirement: JSON-encoded list and object arguments are decoded
Before a tool runs, an argument whose declared type is a list or an object, but whose value is a string, SHALL be decoded if the string holds JSON or a Python literal of exactly that type, and the decoded value SHALL replace the string. Any other argument SHALL be passed through unchanged.

#### Scenario: Stringified list is accepted
- **WHEN** the model calls `format_apa_citation` with `authors` set to the string `["Ada Lovelace", "Alan Turing"]`
- **THEN** the tool receives a list of two names and formats the citation

#### Scenario: Python list literal is accepted
- **WHEN** the model calls `format_apa_citation` with `authors` set to the string `['E. Kirkinis', 'A. V. Andreev']`
- **THEN** the tool receives a list of two names and formats the citation

#### Scenario: Wrong type still fails
- **WHEN** the model passes `authors` as the string `Ada Lovelace` (neither a JSON list nor a Python list literal)
- **THEN** the tool call fails validation and is counted as a tool error

#### Scenario: Strings stay strings
- **WHEN** an argument declared as a string contains text that looks like JSON
- **THEN** the tool receives the original string

### Requirement: Truncated final replies fail the run
If the agent's final reply was cut off by the completion-token limit and contains no tool call, the run SHALL be recorded with status `failed`. Its error SHALL say that the reply hit the token limit. Such a run SHALL NOT be recorded as `completed`.

#### Scenario: Review cut off mid-tool-call
- **GIVEN** the model starts a `write_review` call whose arguments do not fit in the token limit
- **WHEN** the server stops the reply at the limit and the run ends without a parsed tool call
- **THEN** the run is recorded as `failed` with an error naming the token limit

#### Scenario: Normal final answer
- **WHEN** the final reply ends normally (`finish_reason` `stop`)
- **THEN** the run is recorded as `completed`

### Requirement: Repeated identical calls stop the run
When the agent's model requests the same tool call, meaning the same tool with the same arguments, for the `AGENT_MAX_REPEATED_CALLS`-th time in a row (default 8), the run SHALL stop before that call executes and SHALL be recorded with status `loop`. Tool calls made inside a tool SHALL NOT count. A value of 0 SHALL switch this check off.

#### Scenario: Same review requested again and again
- **WHEN** a level-3 agent calls `conduct_literature_review` with the same query seven times in a row and then requests it an eighth time
- **THEN** the eighth call does not execute, and the run is recorded as `loop` with an error naming `conduct_literature_review` and its arguments

#### Scenario: Different arguments
- **WHEN** the agent calls `search_arxiv` ten times in a row with a different query each time
- **THEN** the run is not stopped by this check

#### Scenario: Another call in between
- **WHEN** the agent calls `read_file` with the same path four times, calls `search_arxiv` once, and then calls `read_file` with that path four more times
- **THEN** the run is not stopped by this check

#### Scenario: Several identical calls in one reply
- **GIVEN** a model that may request several tool calls in one reply
- **WHEN** one reply requests the same call eight times
- **THEN** the run stops before any of those calls executes and is recorded as `loop`

#### Scenario: Check switched off
- **WHEN** `AGENT_MAX_REPEATED_CALLS=0` and the agent repeats one call until the step limit
- **THEN** the run is recorded as `step_limit`

### Requirement: A streak of failed tool calls stops the run
When `AGENT_MAX_TOOL_ERROR_STREAK` tool calls of the agent in a row fail (default 10), the run SHALL stop before the next reply of the agent's model and SHALL be recorded with status `loop`. A tool call that succeeds SHALL reset the count. A value of 0 SHALL switch this check off.

#### Scenario: Invented queries that all fail
- **WHEN** a level-3 agent calls `conduct_literature_review` with a different query each time and ten calls in a row fail
- **THEN** the agent's model is not called again, and the run is recorded as `loop` with an error naming the streak and the last failed tool

#### Scenario: A success resets the count
- **WHEN** nine tool calls fail, the next one succeeds and nine more fail
- **THEN** the run is not stopped by this check

#### Scenario: Agent finishes after the streak
- **WHEN** the tenth failed call is the agent's last tool call and its model would not reply again
- **THEN** the run is recorded as `loop`, because the guard stops it before that reply

### Requirement: The step limit still applies
A run that the loop guard does not stop SHALL still end at the step limit with status `step_limit`. When both limits would apply, the first one reached SHALL decide the status.

#### Scenario: Slow loop with varying arguments
- **WHEN** the agent alternates between two calls that succeed until it has made 100 model calls
- **THEN** the run is recorded as `step_limit`

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
