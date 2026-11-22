# Spec Delta

## ADDED Requirements

### Requirement: Caches are filled before an evaluation
The project SHALL provide a `prewarm` command that, for every scenario in a scenarios file, fills the shared caches before any run:
- it runs the topic search that the level-3 tool would issue;
- it looks up the metadata of the returned candidates;
- it downloads their PDFs.

It SHALL report how many entries it added and how many PDFs could not be obtained.

#### Scenario: Prewarm before the final run
- **WHEN** a user runs `prewarm` for the ten base scenarios
- **THEN** the level-3 searches of all ten topics and the PDFs of their candidates are answered from the caches during the evaluation

### Requirement: Missing PDFs are cached
When a PDF download ends with HTTP 404 for every URL tried, the outcome SHALL be stored in the PDF cache as missing. A later download of the same paper SHALL fail at once with the same error, without contacting arXiv. Other failures (timeouts, rate limits, server errors) SHALL NOT be cached.

#### Scenario: Persistent 404
- **GIVEN** the PDF of 2209.14086 returned 404 once
- **WHEN** any later run downloads it
- **THEN** the download fails immediately with the same error, and no HTTP request is made

### Requirement: A missing unversioned PDF is tried at its latest version
When the unversioned PDF URL of a paper returns 404, the download SHALL be retried once at the URL of the paper's latest version, taken from its arXiv metadata. The paper is cached as missing only if that also fails.

#### Scenario: Versioned URL exists
- **WHEN** `https://arxiv.org/pdf/<id>` returns 404 and `https://arxiv.org/pdf/<id>v2` serves the PDF
- **THEN** the download succeeds with the v2 PDF

### Requirement: Cache misses are counted per run
Each run record SHALL count the arXiv API requests and the PDF downloads of that run that were not answered from the cache. Records written before this requirement SHALL report the counts as unknown.

#### Scenario: Fully cached run
- **WHEN** every search, lookup and download of a run is served from the caches
- **THEN** both counts are 0
