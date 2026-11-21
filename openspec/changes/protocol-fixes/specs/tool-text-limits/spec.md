# Spec Delta

## Purpose
Defines how tools shorten text before it reaches a model, so that no level loses part of a paper without the agent knowing, and so that the amount of text each run read can be compared across levels.

## ADDED Requirements

### Requirement: Shortened text carries a visible marker
When a tool returns text to the agent, or passes text to a model, that is shorter than its source, the text SHALL end with a marker that states how many characters were kept and how many the source has. Text that was not shortened SHALL carry no marker.

#### Scenario: Reading a long file
- **WHEN** the agent reads a 66,000-character text file with `read_file` and the limit is 6,000 characters
- **THEN** the result holds the first 6,000 characters followed by a marker stating 6,000 of 66,000 characters

#### Scenario: Short file
- **WHEN** the agent reads a 2,000-character file
- **THEN** the result is the whole file without a marker

### Requirement: Chunk summaries accept at most one chunk
`summarize_chunk` SHALL reject an input file longer than one chunk (`CHUNK_SIZE` characters) with a tool error. The error SHALL tell the agent to split the text into chunks first. Chunk files written by `split_text_into_chunks` SHALL always be accepted.

#### Scenario: Whole paper passed to the chunk summarizer
- **WHEN** the agent calls `summarize_chunk` on `texts/<id>.txt` with 66,000 characters
- **THEN** the call fails with an error that names `split_text_into_chunks`, and no model call is made

#### Scenario: A chunk
- **WHEN** the agent calls `summarize_chunk` on a file produced by `split_text_into_chunks`
- **THEN** the chunk is summarized

### Requirement: Characters read per paper are recorded
Each run record SHALL hold, for every paper whose text reached a model or the agent in that run, the number of distinct characters of its extracted text that were read. Text is read when the agent receives it from a reading tool or when a tool passes it to a model. Records written before this requirement SHALL report the value as unknown.

#### Scenario: Level-2 paper
- **WHEN** `get_and_summarize_paper` summarizes six 3,000-character chunks of a paper (with a 200-character overlap)
- **THEN** the record lists that paper with the number of distinct characters those chunks cover

#### Scenario: Level-1 agent reads a file
- **WHEN** the agent reads the first 6,000 characters of `texts/<id>.txt` and summarizes no chunk
- **THEN** the record lists that paper with 6,000 characters read
