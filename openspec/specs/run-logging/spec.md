# run-logging Specification

## Purpose

Defines what the evaluation log records and where it is kept, so that each run's steps, loops and failures can be followed while an evaluation runs and checked afterwards without MLflow.

## Requirements

### Requirement: Each evaluation logs next to its results file
The `run` command SHALL write its log to a file next to its results file, with the same name and the suffix `.log`. A resumed evaluation SHALL append to that file.

#### Scenario: Log next to the results
- **WHEN** `run --output results/presets-qwen.jsonl` executes
- **THEN** `results/presets-qwen.log` holds the evaluation's log lines

#### Scenario: A resumed evaluation appends
- **GIVEN** `results/presets-qwen.log` holds the lines of an earlier invocation
- **WHEN** the evaluation is resumed with the same output file
- **THEN** the earlier lines are kept and the new lines follow them

### Requirement: Only the run command writes a log file
Only the `run` command SHALL create or write a log file. Importing the package, calling the evaluation from code and the `report` command SHALL NOT write to any log file.

#### Scenario: Tests leave no log behind
- **WHEN** the test suite runs from the repository root
- **THEN** no log file is created or changed outside the tests' temporary directories

#### Scenario: Report
- **WHEN** `report results/presets-qwen.jsonl` runs
- **THEN** its messages go to the console and no log file is written

### Requirement: Lines reach the file as they happen
Each log line SHALL be in the log file as soon as the event it describes has happened, so that a run that is stopped or crashes leaves every line written before.

#### Scenario: Stopped run
- **WHEN** the evaluation process is killed in the middle of a run
- **THEN** the log file holds that run's start line and the step lines of every step that ended before the kill

### Requirement: The run ID is on every line written during a run
Every log line written while a run executes SHALL carry the run's ID, including lines written by services that a tool calls.

#### Scenario: Service warning inside a tool
- **WHEN** an arXiv request made by the level-3 review tool during run `level3-rag-2024h1-01` is retried
- **THEN** the warning line carries `level3-rag-2024h1-01`

### Requirement: The start of an evaluation is logged
When an evaluation starts, the log SHALL get one line naming the model, the levels, the scenarios, the number of repeats, the order seed, the number of runs already finished in the results file and the number of runs still to execute.

#### Scenario: Resumed evaluation
- **WHEN** an evaluation of three levels, one scenario and two repeats is resumed with four finished runs in its results file
- **THEN** its start line says that four runs are finished and two are still to execute

### Requirement: The start of a run is logged
Before the agent of a run starts, the log SHALL get one line with the run ID, the run's position in its block and its seed.

#### Scenario: Run start
- **WHEN** run `level3-rag-2024h1-02` starts as the second of three runs in its block
- **THEN** the log gets a line with its run ID, position 2 of 3 and its seed before any of its step lines

### Requirement: The end of a run is logged
When a run ends, the log SHALL get one line with the run's status, its success, its total tokens, its wall time, its number of tool calls and its number of tool errors.

#### Scenario: Run stopped by the step limit
- **WHEN** a level-3 run hits the step limit after 100 tool calls, 99 of which failed
- **THEN** its end line shows the status `step_limit`, no success, 100 tool calls and 99 tool errors

### Requirement: Each reply of the agent's model gets one line
Each reply of the agent's model SHALL get one line at DEBUG level with the number of tool calls it makes, the number of malformed tool calls in it, its finish reason when that is not a normal stop, and the start of its text, if it has any.

#### Scenario: Reply cut by the token limit
- **WHEN** a reply is cut off by the completion-token limit
- **THEN** its line shows the finish reason `length`

#### Scenario: Call to a tool the level does not offer
- **WHEN** a reply calls a tool name that the run's level does not offer
- **THEN** its line counts one malformed tool call

### Requirement: Each tool call of the agent gets one line
Each tool call of the agent SHALL get one line at DEBUG level with the tool name, its arguments, and either its result or the type and message of its error, together with the call's duration.

#### Scenario: Failed tool call
- **WHEN** `conduct_literature_review` fails because no paper matches its query
- **THEN** its line names the tool, shows the query, and gives the error's type and message

#### Scenario: Successful tool call
- **WHEN** `read_file` returns the content of `reviews/review.md`
- **THEN** its line names the tool, shows the path, and shows the start of the content

### Requirement: Model calls inside a tool are summed up on the tool's line
Model calls made inside a tool SHALL NOT get lines of their own. The line of the tool call SHALL give their number and their summed duration.

#### Scenario: Level-3 review tool
- **WHEN** the level-3 review tool makes 30 model calls during one tool call
- **THEN** the log has one line for that tool call, which states 30 model calls and their summed duration, and no line for each of those calls

### Requirement: Step lines are short
Arguments, results and text in the lines of model replies and tool calls SHALL be shortened to at most 150 characters each, and each such line SHALL be a single line.

#### Scenario: Long review text
- **WHEN** the agent appends 5,000 characters of review text with one tool call
- **THEN** the line of that call shows at most 150 characters of the text and contains no line break

### Requirement: Repeated identical tool calls are flagged
When the agent makes the same tool call, meaning the same tool with the same arguments, for the third time in a row, the log SHALL get one WARNING line that names the call. This warning SHALL be written at most once per run.

#### Scenario: Same review requested again and again
- **WHEN** a level-3 agent calls `conduct_literature_review` with the same query three times in a row and then keeps doing so
- **THEN** the log gets one warning naming the call after the third call, and the later identical calls in that run add no further warning

#### Scenario: Different arguments
- **WHEN** the agent calls one tool three times in a row with different arguments each time
- **THEN** no repeated-call warning is written

### Requirement: Consecutive tool errors are flagged
When five tool calls of the agent in a row end in an error, the log SHALL get one WARNING line that names the tool of the fifth failed call. This warning SHALL be written at most once per run.

#### Scenario: Invented queries that all fail
- **WHEN** a level-3 agent calls `conduct_literature_review` with a different query each time and every call fails
- **THEN** the log gets one warning after the fifth failed call, and no further error warning in that run

#### Scenario: A success resets the count
- **WHEN** four tool calls fail, the next one succeeds and four more fail
- **THEN** no error warning is written

### Requirement: The console shows less than the file
The console SHALL show log lines at INFO level and above unless `TOOL_EVAL_LOG_LEVEL` sets another level. The log file SHALL keep lines at DEBUG level and above.

#### Scenario: Default settings
- **WHEN** an evaluation runs with default settings
- **THEN** the console shows the start and end lines and the warnings but no step lines, and the log file holds all of them

#### Scenario: Steps on the console
- **WHEN** an evaluation runs with `TOOL_EVAL_LOG_LEVEL=DEBUG`
- **THEN** the console also shows the step lines

### Requirement: Logging never changes runs or results
Logging SHALL NOT change the agent's behaviour, the run records or the configuration hash. A failure while writing a log line SHALL NOT stop a run.

#### Scenario: Results written before this change
- **WHEN** a results file written before this change is resumed
- **THEN** its finished runs are recognised and skipped, because the configuration hash is unchanged

#### Scenario: A warning does not stop the run
- **WHEN** a loop warning is written
- **THEN** the run continues until the agent finishes, the loop guard stops it or the step limit ends it

#### Scenario: Logging fails
- **WHEN** writing a step line raises an error
- **THEN** the run continues and its record is written as usual

### Requirement: A run stopped by the loop guard is logged
When the loop guard stops a run, the run's end line SHALL show the status `loop`, and the log SHALL get one WARNING line that names the signal and the call or tool that triggered it.

#### Scenario: Repeated review call
- **WHEN** the guard stops a run before the eighth identical `conduct_literature_review` call
- **THEN** the log has a warning naming that call and an end line with status `loop` and no success
