# Spec Delta

## ADDED Requirements

### Requirement: Searches are ordered by relevance
arXiv searches SHALL return the most relevant papers within the scenario's submission-date window first, not the newest submissions. Within a scenario, results SHALL stay the same across runs and models; the on-disk cache provides this.

#### Scenario: Topical results first
- **WHEN** an agent searches `protein structure prediction` within 2022-07-01..2023-06-30
- **THEN** the results are ranked by arXiv's relevance to the query, not by submission date

#### Scenario: Same results for every model
- **WHEN** two models issue the same search in the same scenario
- **THEN** they receive the same papers in the same order

### Requirement: PDFs up to 100 MB can be downloaded
Paper PDFs SHALL be downloadable up to 100 MB, a limit separate from the limit for workspace text files. A PDF over the limit, or one that arXiv does not serve, SHALL fail with an error that names the paper and the reason (size limit or HTTP status).

#### Scenario: Large paper
- **WHEN** a tool downloads a 46.5 MB PDF (for example `2312.09754`)
- **THEN** the download succeeds and the paper can be read

#### Scenario: Oversized paper
- **WHEN** a PDF exceeds 100 MB
- **THEN** the download fails with an error naming the paper and the 100 MB limit

#### Scenario: Missing PDF
- **WHEN** arXiv answers 404 for a paper's PDF (for example `2301.00016`)
- **THEN** the download fails with an error naming the paper and the HTTP status
