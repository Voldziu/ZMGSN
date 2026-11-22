# Spec Delta

## ADDED Requirements

### Requirement: MLflow parameters describe the sampling and the served model
When tracing is enabled, each evaluation run's MLflow parameters SHALL include:
- `top_p`, `top_k` and the completion token limit;
- the served model's checkpoint, quantisation, tool-call parser, chat-template hash and vLLM version.

These SHALL be the same values that enter the configuration hash.

#### Scenario: Comparing qwen-mid servers
- **WHEN** runs of `qwen-mid` are traced against a BF16 server
- **THEN** their MLflow parameters show the checkpoint `Qwen/Qwen3.5-4B` with no quantisation
