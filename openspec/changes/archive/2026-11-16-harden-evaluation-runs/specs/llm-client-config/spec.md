# Spec Delta

## ADDED Requirements

### Requirement: Each run samples with its own recorded seed
Each evaluation run SHALL draw a fresh random sampling seed. It SHALL send that seed with every model request made during the run, both the agent's own requests and those made inside tools. The run record SHALL store the seed. The seed SHALL NOT be part of the configuration hash.

#### Scenario: Repeats differ across server restarts
- **GIVEN** a preset server is restarted between two sessions
- **WHEN** repeat 1 of the same scenario and level runs in each session
- **THEN** the two runs use different seeds, and their records show both seeds

#### Scenario: Tool-internal requests share the run's seed
- **WHEN** a level-3 tool calls the model while summarising papers
- **THEN** those requests carry the same seed as the agent's requests in that run

#### Scenario: Seed does not split comparable runs
- **WHEN** two runs differ only in their seeds
- **THEN** their configuration hashes are equal

## MODIFIED Requirements

### Requirement: Results record the settings that change them
Each run record SHALL carry a configuration hash that covers:
- the step limit;
- the effective parallel-tool-call policy;
- the version of the tool definitions;
- the model and prompt settings.

Runs whose hashes differ SHALL NOT be treated as comparable.

#### Scenario: Step limit changes the hash
- **WHEN** two runs differ only in `AGENT_MAX_STEPS`
- **THEN** their configuration hashes differ

#### Scenario: Parallel-call policy changes the hash
- **WHEN** two runs of `qwen` differ only in `LLM_PARALLEL_TOOL_CALLS`
- **THEN** their configuration hashes differ

#### Scenario: Tool definitions change the hash
- **WHEN** the tools of any level change (for example, level 2 replacing `append_to_review` with `write_review`)
- **THEN** runs made before and after the change have different configuration hashes
