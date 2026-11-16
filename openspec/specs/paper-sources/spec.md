# paper-sources Specification

## Purpose
Defines how the tools and the evaluation obtain papers. This covers arXiv search and metadata lookups, which must be gentle on arXiv and repeatable across runs, and turning a downloaded PDF into text the tools can use.

## Requirements

### Requirement: arXiv responses are cached on disk
Every arXiv API response that succeeds, whether from a search or an ID lookup (including the evaluation's own checks), SHALL be stored in an on-disk cache shared by all runs. A later request with the same parameters SHALL be answered from the cache without contacting arXiv. Failed responses SHALL NOT be cached.

#### Scenario: Repeated search is served from cache
- **GIVEN** a search for `retrieval augmented generation` within 2024-01-01..2024-06-30 has succeeded once
- **WHEN** any later run issues the same search
- **THEN** it gets the same papers without an HTTP request to arXiv

#### Scenario: Cache survives a restart
- **WHEN** the evaluation is started again in a new process
- **THEN** responses cached by earlier processes are still used

#### Scenario: Failures are not cached
- **WHEN** an arXiv request ends in an error after all retries
- **THEN** nothing is written to the cache, and the next identical request contacts arXiv again

### Requirement: arXiv requests are spaced and retried
Requests that reach arXiv SHALL be at least 3 seconds apart. A request answered with HTTP 429, a server error or a timeout SHALL be retried with increasing waits that honour a `Retry-After` header. Only when the retries are exhausted SHALL the request fail with an external-service error.

#### Scenario: Rate limit is ridden out
- **WHEN** arXiv answers 429 once and then succeeds
- **THEN** the request returns the papers, after waiting at least the `Retry-After` time when one is given

#### Scenario: Persistent outage
- **WHEN** arXiv keeps answering 429 for every retry
- **THEN** the request fails with an external-service error that names arXiv

#### Scenario: Client errors are not retried
- **WHEN** arXiv answers 400 to a malformed query
- **THEN** the request fails at once, without retries

### Requirement: Extracted PDF text is always valid UTF-8
Text extracted from a PDF SHALL be stored and passed on as valid UTF-8. Characters that cannot be encoded, such as lone UTF-16 surrogates, SHALL be replaced instead of failing the tool call. The PDF reader SHALL be able to decode the CFF Type 1 fonts common in LaTeX papers.

#### Scenario: Surrogates in extracted text
- **WHEN** a PDF yields text that contains a lone surrogate character
- **THEN** the text file is written with that character replaced, and the paper can be summarised

#### Scenario: LaTeX fonts
- **WHEN** a paper typeset with Computer Modern (CFF Type 1) fonts is extracted
- **THEN** the reader does not warn that a font-parsing library is missing

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
