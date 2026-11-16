# Spec Delta

## ADDED Requirements

### Requirement: Level-3 review fills the requested number of read papers
The level-3 review tool SHALL search more candidates than the review needs. It SHALL include papers in search order, skipping any it cannot read, until the review holds the requested number of read papers or the candidates run out. It SHALL never include a paper it did not read. It SHALL report which candidates it skipped and why.

#### Scenario: One candidate cannot be read
- **GIVEN** a review of 5 papers is requested and the third candidate's PDF is unavailable
- **WHEN** the level-3 tool runs
- **THEN** the review contains 5 read papers, and the skipped candidate is reported with its reason

#### Scenario: Too few readable candidates
- **WHEN** fewer than 5 of all candidates can be read
- **THEN** the review contains only the papers that were read, and the tool reports the shortfall
