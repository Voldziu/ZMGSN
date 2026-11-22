# Spec Delta

## ADDED Requirements

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
