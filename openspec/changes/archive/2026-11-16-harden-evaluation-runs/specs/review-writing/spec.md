# Spec Delta

## Purpose

Defines how the agent writes the review file `reviews/review.md` at each tool level. Rewriting the review must replace earlier text rather than stack on top of it.

## ADDED Requirements

### Requirement: Level-2 review writing replaces the file
At level 2, the agent SHALL write the review with one tool, `write_review`, that replaces the whole content of `reviews/review.md` with the Markdown it is given. Level 2 SHALL NOT offer a tool that appends to the review.

#### Scenario: Rewriting does not duplicate
- **WHEN** the agent calls `write_review` with a five-section review and later calls it again with a corrected five-section review
- **THEN** `reviews/review.md` contains only the corrected review, with five sections

#### Scenario: First write creates the file
- **WHEN** `reviews/review.md` does not exist and the agent calls `write_review`
- **THEN** the file is created with exactly the given Markdown

### Requirement: Level-2 bibliography replaces the previous one
At level 2, the bibliography tool SHALL write a `## Bibliography` section with sorted APA references at the end of the review. If such a section already exists, it SHALL be replaced rather than duplicated. The text before it SHALL be left unchanged.

#### Scenario: Bibliography written twice
- **WHEN** the agent calls the bibliography tool twice with different identifiers
- **THEN** the review contains one `## Bibliography` section, which lists the identifiers of the second call

#### Scenario: Review rewritten after the bibliography
- **WHEN** the agent calls `write_review` after writing the bibliography
- **THEN** the file contains exactly the new Markdown, and the agent must write the bibliography again to have one
