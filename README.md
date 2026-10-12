# tool_eval

How tool granularity (`level1`, `level2`, `level3`) affects an LLM research-assistant agent.

## Setup

```sh
uv sync
cp .env.example .env
```

Start the LM Studio local server (default `http://localhost:1234/v1`), load a model and set `LLM_MODEL` in `.env`.

## Evaluate

```sh
uv run python -m tool_eval.main run                          # all scenarios, all levels, 10 repeats
uv run python -m tool_eval.main run --only rag-2024h1 --levels level1 --repeats 1   # quick check
uv run python -m tool_eval.main report                       # summary of results/results.jsonl
```

Scenarios are in `scenarios/scenarios.json`. Run options: `--scenarios`, `--only`, `--levels`, `--repeats`, `--temperature`, `--output`, `--runs-dir`.

## Checks

```sh
uv run pytest
```
