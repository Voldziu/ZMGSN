# Spec Delta

## Purpose

Defines the order in which an evaluation executes its runs, so that no tool level systematically runs first in a block of one repeat and one scenario and pays for cold caches, while the order stays reproducible from a recorded seed.

## ADDED Requirements

### Requirement: Level order is randomised within each block
An evaluation SHALL run its runs repeat by repeat, and each scenario's levels within a repeat as one block. Each block SHALL have a random planned order over all levels of the tool set, determined by the order seed, the repeat and the scenario, so that the same inputs always give the same order. The evaluation SHALL run the levels it was given in their planned order.

#### Scenario: Orders differ between blocks
- **WHEN** an evaluation runs three levels on ten scenarios
- **THEN** every block runs all three levels, and not every block uses the same order

#### Scenario: Same seed, same order
- **WHEN** two evaluations use the same order seed, scenarios and levels
- **THEN** they run the levels of every block in the same order

#### Scenario: Some of the levels
- **WHEN** an evaluation runs only levels 1 and 3
- **THEN** each block runs them in the order they have in its planned order over all three levels

#### Scenario: Interrupted evaluation is balanced
- **WHEN** an evaluation is interrupted during repeat 2
- **THEN** every scenario has finished repeat 1 at every level

### Requirement: The order seed is recorded
Each run record SHALL store the order seed of its evaluation and the run's 1-based position in its block's planned order over all levels of the tool set, so that positions within a block stay unique when a results file is resumed with other levels. The order seed SHALL NOT be part of the configuration hash.

#### Scenario: Order can be recomputed
- **WHEN** a run is recorded
- **THEN** its record holds the order seed and its position in the block, and the planned order of the block can be recomputed from the seed alone

#### Scenario: Resumed with more levels
- **GIVEN** a results file whose blocks ran only level 1
- **WHEN** the evaluation is resumed with all three levels
- **THEN** the runs of each block have different positions, which follow the block's planned order

#### Scenario: Retried run
- **WHEN** a run that ended with an infrastructure error is run again later
- **THEN** its record keeps its planned position, and its start time shows when it ran

#### Scenario: Seed does not split comparable runs
- **WHEN** two evaluations differ only in their order seeds
- **THEN** their records have equal configuration hashes

### Requirement: The order seed can be fixed and survives a resume
The `run` command SHALL accept an order seed. Without one, an evaluation SHALL reuse the order seed recorded in its results file, or draw a new random seed if the file holds none.

#### Scenario: Fixed seed
- **WHEN** a user runs `run --order-seed 7`
- **THEN** every record of that evaluation stores order seed 7

#### Scenario: Resume keeps the plan
- **GIVEN** a results file whose records hold order seed 7
- **WHEN** the evaluation is resumed without `--order-seed`
- **THEN** the remaining runs use seed 7 and follow the planned order of their blocks

#### Scenario: New results file
- **WHEN** an evaluation starts a new results file without `--order-seed`
- **THEN** it draws a random order seed and records it
