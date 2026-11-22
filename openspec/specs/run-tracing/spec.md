# run-tracing Specification

## Purpose
Records every evaluation run in a self-hosted MLflow tracking server: its settings, results and review as an MLflow run, and every model and tool call as one trace. Runs can then be browsed, filtered and diagnosed without ad-hoc scripts.

## Requirements

### Requirement: Self-hosted tracking server
The project SHALL provide an MLflow tracking server that runs in Docker Compose, separately from the model presets. Its version SHALL match the client library. It SHALL keep its data on a Docker volume across restarts. It SHALL listen on port 5050 on all interfaces, so that its owner can also reach it from other devices on the local network. Each user SHALL run their own server.

#### Scenario: Start the server
- **WHEN** a user runs `make mlflow-up`
- **THEN** the tracking server starts, its `/health` endpoint answers `OK`, and the UI is available at `http://localhost:5050`

#### Scenario: Reachable from the local network
- **GIVEN** the server runs on a machine with LAN address `192.168.1.4`
- **WHEN** another device of its owner on the network requests `http://192.168.1.4:5050/health`
- **THEN** it answers `OK`

#### Scenario: Presets do not stop the server
- **GIVEN** the tracking server is running
- **WHEN** `make eval-presets` starts and stops model presets
- **THEN** the tracking server keeps running

#### Scenario: Data survives a restart
- **WHEN** the server is stopped with `make mlflow-down` and started again
- **THEN** earlier runs and traces are still listed

### Requirement: Tracing is opt-in and fails fast
Evaluation runs SHALL be recorded in MLflow when `MLFLOW_TRACKING_URI` is set, and only then. When it is set, the evaluation SHALL check the server's health before the first run. If the server is not healthy, the evaluation SHALL stop with an error that names the URI, without starting any run.

#### Scenario: Tracing disabled
- **WHEN** `MLFLOW_TRACKING_URI` is not set
- **THEN** runs execute and are recorded in the results file without contacting any MLflow server

#### Scenario: Server down
- **WHEN** `MLFLOW_TRACKING_URI=http://localhost:5050` is set and no server answers
- **THEN** the evaluation exits with an error naming the URI before any agent run starts

### Requirement: Each evaluation run is an MLflow run
With tracing enabled, each evaluation run SHALL be logged as one MLflow run in the configured experiment (default `tool-eval`). The MLflow run SHALL carry:
- **parameters:** model, level, scenario, repeat, seed, order seed, position in the block, step limit, loop-guard limits, temperature, prompt version, tool-set version, checks version and config hash;
- **metrics:** score, success (under the success rule), model calls, tool calls, tool errors, malformed tool calls, tokens (including the total and the first agent call's input tokens), model and tool time, and wall time;
- **tags:** status and error;
- **artifacts:** the review file and the run record.

A measurement that the run's record does not know SHALL NOT be logged.

#### Scenario: Compare models in the UI
- **WHEN** `qwen` and `llama` were evaluated with tracing enabled
- **THEN** their runs appear in one experiment with `model` and `level` parameters and a `score` metric, so that they can be filtered and compared

#### Scenario: Review attached
- **WHEN** a run wrote `reviews/review.md`
- **THEN** the file is stored as an artifact of the run's MLflow run

#### Scenario: Step-limit run is not a success
- **WHEN** a run that hit the step limit passed every check
- **THEN** its `success` metric is 0 and its `score` metric is 1

#### Scenario: Prompt overhead and time
- **WHEN** a run is logged
- **THEN** its metrics include the total tokens, the first agent call's input tokens, the agent and tool-internal model time and the tool time

#### Scenario: Loop run
- **WHEN** the loop guard stopped a run
- **THEN** its `status` tag is `loop`, its `error` tag names the signal, and its `success` metric is 0

### Requirement: One trace per run holds every call
With tracing enabled, each evaluation run SHALL produce exactly one trace, linked to its MLflow run. The trace SHALL contain every agent model call, every tool call, and every model call made inside a tool, nested under the tool that made it. It SHALL record token usage per model call, and it SHALL mark failed tool calls as errors that carry the exception type and message.

#### Scenario: Nested tool-internal calls
- **WHEN** a level-3 run's review tool summarises papers with the model
- **THEN** those model calls appear in the trace as children of the review tool's span

#### Scenario: Failed tool call
- **WHEN** a tool call fails, for example with a validation error or "File not found"
- **THEN** its span has error status and records the exception type and message, while the agent continues

#### Scenario: Filter traces
- **WHEN** a user filters traces by the tag `level = 'level1'` or by the session `<model>/<scenario>`
- **THEN** only the matching runs' traces are listed

### Requirement: Scores are attached to traces
With tracing enabled, the outcome score and the result of every criterion SHALL be attached to the run's trace as code-sourced assessments. The run record in the results file SHALL store the MLflow run ID and the trace ID.

#### Scenario: Criteria visible on the trace
- **WHEN** a scored run's trace is opened
- **THEN** it shows the `score` and one assessment per criterion, including `papers_read`, with its pass or fail value and detail

#### Scenario: Link from results to trace
- **WHEN** a run record is read from the results file
- **THEN** its MLflow run ID and trace ID identify the matching run and trace on the server

### Requirement: MLflow parameters describe the sampling and the served model
When tracing is enabled, each evaluation run's MLflow parameters SHALL include:
- `top_p`, `top_k` and the completion token limit;
- the served model's checkpoint, quantisation, tool-call parser, chat-template hash and vLLM version.

These SHALL be the same values that enter the configuration hash.

#### Scenario: Comparing qwen-mid servers
- **WHEN** runs of `qwen-mid` are traced against a BF16 server
- **THEN** their MLflow parameters show the checkpoint `Qwen/Qwen3.5-4B` with no quantisation
