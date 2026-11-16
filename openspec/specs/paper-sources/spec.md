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
