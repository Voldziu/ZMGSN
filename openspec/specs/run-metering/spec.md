# run-metering Specification

## Purpose
Defines what is measured during each evaluation run, so that cost, time and stability can be compared across tool levels and models: model calls and tokens of the agent and of tools, tool calls and errors, malformed tool calls, model and tool time, wall time and the prompt overhead.

## Requirements

### Requirement: Model calls and tokens are attributed to the agent or a tool
Each run record SHALL count the model calls and the input and output tokens reported by the server, separately for calls of the agent loop and for calls made inside a tool. The run's total tokens SHALL include the tool-internal calls. A model call that reports no token usage SHALL be counted as a call without usage.

#### Scenario: Tool-internal calls are part of the cost
- **WHEN** a level-3 run's review tool summarises papers with the model
- **THEN** those calls and their tokens are counted as tool-internal, and the run's total tokens include them

#### Scenario: Agent calls
- **WHEN** the agent loop calls the model
- **THEN** the call and its tokens are counted for the agent

### Requirement: Tool calls and tool errors are counted
Each run record SHALL count every tool run as a tool call, and every tool run that ends with an exception as a tool error, including argument validation failures.

#### Scenario: Failed tool call
- **WHEN** a tool call fails validation and the agent continues
- **THEN** the record counts one tool call and one tool error

### Requirement: Model and tool time are measured
Each run record SHALL store three summed durations: the agent's model calls, the model calls made inside tools, and the tool calls themselves, including the model calls made inside them. Durations of calls that overlap in time SHALL be added.

#### Scenario: Time inside a level-3 tool
- **GIVEN** a level-3 run whose review tool runs for 60 s, of which its model calls take 40 s
- **WHEN** the run is recorded
- **THEN** its tool time is 60 s and its tool-internal model time is 40 s, which leaves 20 s of other tool time

#### Scenario: Agent model time
- **WHEN** the agent loop makes two model calls of 3 s each
- **THEN** the record's agent model time is 6 s

### Requirement: Wall time excludes scoring
The wall time of a run SHALL cover building the agent and the agent loop. It SHALL NOT include scoring the review or opening the run's MLflow run.

#### Scenario: Slow scoring
- **GIVEN** scoring a review takes 30 s because the arXiv lookups are throttled
- **WHEN** the run is recorded
- **THEN** its wall time does not include those 30 s

#### Scenario: Slow tracking server
- **GIVEN** opening the run's MLflow run takes 10 s
- **WHEN** the run is recorded
- **THEN** its wall time does not include those 10 s

### Requirement: The first agent call's input tokens are recorded
Each run record SHALL store the input tokens reported for the run's first agent model call. This value measures the prompt overhead of the level: system prompt, task and tool schemas. If that call reports no token usage, the value SHALL be unknown.

#### Scenario: Later calls do not change it
- **WHEN** the first agent call of a run reports 5,000 input tokens and later calls report more
- **THEN** the record's first-call input tokens are 5,000

#### Scenario: No usage reported
- **WHEN** the first agent call reports no token usage
- **THEN** the record's first-call input tokens are unknown (`null`), not zero

### Requirement: Malformed tool calls are counted
Each run record SHALL count the agent's tool calls that never start a tool run, per kind: calls whose arguments could not be parsed, and calls to a tool name that the run's level does not offer. Neither kind SHALL count as a tool call or a tool error. The counts SHALL also be kept for runs that end at the step limit or fail.

#### Scenario: Tool of another level
- **GIVEN** a level-3 run
- **WHEN** the agent calls `search_arxiv`, a level-1 tool
- **THEN** the record counts one call to an unknown tool and no tool call

#### Scenario: Unparseable arguments
- **WHEN** the agent's reply holds a tool call whose arguments are not valid JSON
- **THEN** the record counts one unparseable tool call

#### Scenario: Step-limit run
- **WHEN** a run that made malformed tool calls ends at the step limit
- **THEN** its record still holds the counts

### Requirement: Older records report new measurements as unknown
Records written before the model time, tool time, first-call input tokens and malformed-call counts existed SHALL be read with those measurements unknown (`null`), not zero.

#### Scenario: Pilot results file
- **WHEN** a results file written before these measurements existed is read
- **THEN** its records show those measurements as unknown, and their tokens and calls are unchanged

### Requirement: Tool calls returned as text are counted
Each run record SHALL count the agent's replies that carry no parsed tool call, but whose text has the shape of a tool call in any format a served model uses:
- a JSON object with a `name` field and a `parameters` or `arguments` field;
- a `<tool_call>` block;
- a `<function=…>` tag.

These replies SHALL be counted as `text_tool_calls`, separately from unparseable and unknown tool calls. Records written before this requirement SHALL report the count as unknown.

#### Scenario: Parser failure on a long call
- **WHEN** the agent's reply text is `{"name": "write_review", "parameters": {"markdown": "# …` and the reply has no tool calls
- **THEN** the record counts one text tool call

#### Scenario: Final answer
- **WHEN** the agent ends with the plain text "The review is saved in reviews/review.md."
- **THEN** no text tool call is counted

### Requirement: Tool errors are split by cause
Each run record SHALL split its tool errors into:
- **environment errors**: an external service failed, for example a network error, an HTTP error from arXiv or a missing PDF;
- **agent errors**: every other failed tool call, for example invalid arguments, an unknown path, or input the tool rejects.

The two counts SHALL add up to the total number of tool errors. Records written before this requirement SHALL report both counts as unknown.

#### Scenario: Missing PDF
- **WHEN** the agent downloads a paper whose PDF is missing on arXiv
- **THEN** the call counts as one environment error

#### Scenario: Invalid argument
- **WHEN** the agent passes a malformed arXiv identifier
- **THEN** the call counts as one agent error
