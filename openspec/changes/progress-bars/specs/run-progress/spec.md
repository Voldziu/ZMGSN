# Spec Delta

## Purpose

Defines the progress bars an evaluation shows on the console: which levels of the plan they track and when they are drawn.

## ADDED Requirements

### Requirement: Nested progress bars on a terminal
While the `run` command executes and stderr is a terminal, the console SHALL show four bars: the model's position among the presets, the model's runs (finished out of planned, with an estimated time left), the current repeat out of all repeats, and the current scenario with the running level.

#### Scenario: Second preset of five
- **GIVEN** `eval-presets.sh` runs the presets `qwen gemma llama qwen-mid qwen-weak`
- **WHEN** gemma's evaluation is running its third repeat, its fourth scenario `rlhf-2024`, level 2
- **THEN** the bars show `2/5 gemma`, the runs of gemma, `3/10` repeats and `rlhf-2024` with `level2`

#### Scenario: Resumed evaluation
- **WHEN** an evaluation of 90 planned runs is resumed with 30 runs already finished
- **THEN** the runs bar starts at 30 of 90

#### Scenario: Run started by hand
- **WHEN** `run` is started without `TOOL_EVAL_PRESET_POSITION`
- **THEN** the model bar shows `1/1` and the model name

### Requirement: No bars without a terminal
When stderr is not a terminal, the evaluation SHALL NOT draw progress bars, and its console output SHALL be the same as without this capability.

#### Scenario: Output piped to a file
- **WHEN** `make eval-presets ... 2>&1 | tee logs/screen.out` runs
- **THEN** the file holds the log lines and no progress-bar output

### Requirement: Log lines do not break the bars
While the bars are drawn, each console log line SHALL appear whole above the bars, and the bars SHALL be redrawn below it. The log file SHALL be unchanged.

#### Scenario: A warning during a run
- **WHEN** a loop warning is logged while the bars are drawn
- **THEN** the warning appears as one line above the bars, and the bars stay below it
