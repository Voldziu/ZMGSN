# Spec Delta

## ADDED Requirements

### Requirement: The quality table counts runs by how they ended
For each model and level, the quality table SHALL show how many scored runs ended at the step limit, how many the loop guard stopped, and how many failed inside the agent, each in its own column.

#### Scenario: Loop runs
- **WHEN** qwen-weak at level 3 has 10 scored runs, of which 7 were stopped by the loop guard and 1 hit the step limit
- **THEN** its row shows 7 in the `loop` column and 1 in the `step limit` column

#### Scenario: Older results
- **WHEN** a results file written before the loop guard is reported
- **THEN** every row shows 0 in the `loop` column
