# task-variants Specification

## Purpose
Defines the variant tasks used to measure flexibility (H1b). Each variant is a base scenario plus one requirement that the base workflow does not cover. This spec covers how variants are defined, how their prompt is built and how runs record them.

## Requirements

### Requirement: Every base scenario has one variant with an added requirement
For every base scenario, the scenarios file SHALL hold one variant with:
- the same topic and date window;
- the id `<base-id>-<requirement>`;
- a reference to its base scenario;
- one requirement, `table`, `polish` or `no_surveys`.

A variant SHALL NOT have a variant of its own.

#### Scenario: Requirement meanings
- **WHEN** a variant requires `table`, `polish` or `no_surveys`
- **THEN** it asks, respectively, for a comparison table of the selected papers under the main heading, for every summary in Polish, or for no cited survey or review paper

#### Scenario: Invalid variant
- **WHEN** a variant refers to a base id that does not exist, or to another variant
- **THEN** loading the scenarios fails with an error that names the variant

### Requirement: Requirements are assigned in turn
The requirements SHALL be assigned in turn over the base scenarios in file order: `table`, `polish`, `no_surveys`, `table`, and so on.

#### Scenario: Ten base scenarios
- **WHEN** the scenarios file holds ten base scenarios
- **THEN** it also holds ten variants, of which 4 require `table`, 3 `polish` and 3 `no_surveys`, the first base scenario's variant being `table`

### Requirement: The variant prompt adds exactly one requirement line
The task prompt of a variant SHALL be the base task prompt for the same topic and window, followed by one requirement statement taken from a versioned prompt file for that requirement. A base task's prompt SHALL contain no requirement statement. The system prompt and the tools SHALL be the same for base tasks and variants.

#### Scenario: Polish variant
- **WHEN** the prompt of `gnn-2023h1-polish` is rendered
- **THEN** it equals the prompt of `gnn-2023h1` plus a statement that every summary must be written in Polish

#### Scenario: Same tools
- **WHEN** a variant runs at level 3
- **THEN** the agent has only the level-3 tool, as on the base task

### Requirement: Run records identify the task type
Every run record SHALL carry the task type (`base` or `variant`), the requirement (none for base tasks) and the id of the base scenario. A base run's base scenario is the scenario itself. Records written before this requirement SHALL load as base tasks.

#### Scenario: Variant record
- **WHEN** `rag-2024h1-table` runs at level 2
- **THEN** its record has task type `variant`, requirement `table` and base scenario `rag-2024h1`

#### Scenario: Old record
- **WHEN** a results file written before this change is read
- **THEN** its records are base tasks with no requirement

### Requirement: Variants are scheduled like base scenarios
Variant scenarios SHALL be planned, ordered and resumed in the same way as base scenarios: one block per scenario and repeat, with the levels in a random order within the block. Selecting scenarios by id SHALL work for variants as well.

#### Scenario: Only variants
- **WHEN** an evaluation is started with only the ten variant ids
- **THEN** it runs the ten variants at every given level, and no base scenario
