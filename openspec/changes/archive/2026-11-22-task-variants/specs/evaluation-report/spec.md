# Spec Delta

## ADDED Requirements

### Requirement: Base tasks and variants are reported separately
The report SHALL summarise base tasks and variant tasks in separate sections and SHALL NOT pool them in any metric. Within the variant section it SHALL also break down the success rate by requirement. Level differences SHALL be computed within one task type only, pairing scenarios of that type.

#### Scenario: Results with both task types
- **WHEN** a results file holds base and variant runs of `qwen`
- **THEN** the report shows one quality and cost table for base tasks and one for variants, plus success by requirement for the variants

#### Scenario: Level 2 against level 3 on variants
- **WHEN** levels 2 and 3 of a model both have variant runs on ten variants
- **THEN** the report shows the variant `level3 − level2` difference computed over those ten variants only

#### Scenario: Only base runs
- **WHEN** a results file holds no variant runs
- **THEN** the report has no variant section
