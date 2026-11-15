# Design

## Context

See proposal.md for the motivation. The evidence comes from 2026-11-15 test runs on the RTX 5080 (WSL2, driver 617.14) and on the Apple M5 Pro (24 GB). The scenario was `rag-2024h1`, level 1, with 3 repeats per preset and step limit.

| Preset | 10 steps | 30 steps |
| --- | --- | --- |
| `qwen` | 3/3 completed, 1.0 (8–9 calls) | 3/3 completed, 1.0 |
| `qwen-mid` | 2/3 completed 1.0, 1/3 step_limit | 3/3 completed, 1.0 (8 calls) |
| `gemma` | 3/3 step_limit, 0.0 (one tool per turn) | 3/3 completed, 1.0 (25–29 calls) |
| `qwen-weak` | 2/3 completed (0.78, 0.44), 1/3 step_limit | 0.22; step_limit; context overflow |
| `llama` | 2× HTTP 400 after turn 1, 1× text-only answer | 3× HTTP 400 after turn 1 |

## Goals / Non-Goals

**Goals:**
- Every preset starts on the RTX 5080 and can finish the level-1 scenario.
- The step limit stops runaway loops but does not penalise models that call one tool per turn.
- Make the minimum adjustments needed for model-specific format quirks, and record each one.

**Non-Goals:**
- Prompt tuning per model.
- Changing the vendored Llama 3.1 chat template.
- Per-run traces of tool calls. Useful for diagnosis, but a separate change.

## Decisions

### CUDA `devel` base image with nvcc/crt/nvvm pinned to 13.0.88
The slim `base` image with pip's CUDA failed in two ways:
- `humming-kernels[cu13]` pulls `nvidia-cuda-nvcc` unpinned, so the compiler (13.5) did not match torch's 13.0 headers.
- FlashInfer links against `$CUDA_HOME/lib64` and `lib64/stubs`, which the pip layout does not have.

The `devel` image is about 16 GB instead of 10 GB, but it ships the toolkit layout FlashInfer expects. The pins stay so that a pip `nvcc` can never mix versions again.
*Alternative:* add symlinks to the slim image. Rejected because other JIT kernels link further libraries (`cublas`, `cublasLt`) the same way.

### `max-num-seqs: 8` on every CUDA preset
The evaluation is strictly sequential. vLLM's default of 256 made CUDA-graph capture fail for `qwen-weak` (142 Mamba cache blocks at 4 GB), and it reserves graph memory the tight presets need.

### `qwen-mid` budget 0.5 (8 GB)
At 0.375, the FP8 model alone took 4.71 GiB, because embeddings and `lm_head` for a vocabulary of about 248k stay in BF16. That left -0.02 GiB of KV cache.
*Alternatives:*
- `sanskar003/Qwen3.5-4B-AWQ`: gated, and a different checkpoint from the Metal preset.
- A shorter context: breaks the 32k parity across presets.

### Parallel tool calls: off for `llama` only, via `parallel_tool_calls: false`
- `llama` emits 5–10 tool calls in its first reply, often with invented arguments. The Llama 3.1 template then rejects that assistant message on the next request (HTTP 400).
- vLLM honours `parallel_tool_calls: false` by returning only the first call. On the RTX this gave 1 call per reply in 5 out of 5 attempts, against 3–6 without the flag.
- The policy is a setting, `LLM_PARALLEL_TOOL_CALLS` (unset = preset default). The preset default is `false` for `llama` and server default for the rest. The decision follows the served model name, because that name is already the contract between client and server.
- Other presets keep parallel calls, so that each model is measured in its natural tool-calling style. The effective value is part of the config hash.

*Alternative:* off for all models. Rejected by the user: it changes `qwen`/`qwen-mid` behaviour (about 23 calls in 8 turns), and the 30-step limit already absorbs one-call-per-turn models.

### Decode JSON-encoded list/object arguments before validation
- Llama 3.1 through the `llama3_json` parser sends nested values as strings, for example `"authors": "[\"A\", \"B\"]"`. Pydantic converts scalar strings such as `"5"` on its own, but rejects strings for lists and objects.
- A tool-call middleware runs before the tool executes. For each argument whose JSON schema type is `array` or `object` and whose value is a string, it tries `json.loads`, and keeps the decoded value only if it has the declared type.
- Anything else is left untouched, so genuinely wrong arguments still fail and still count as tool errors.
- This is applied to every model, because it removes a serialisation quirk, not a reasoning mistake.

### Default step limit 30
- 30 lets one-tool-per-turn models finish (`gemma` used 25–29 calls) and does not change parallel callers (`qwen` stays at 8–9).
- Runaway models still stop: `qwen-weak` hit either the limit or the 32k context, and both are recorded as failures.
- `AGENT_MAX_STEPS` still overrides the default.

## Risks / Trade-offs

- [Results made before this change are not comparable.] → `max_steps` and the parallel-call policy are in the config hash, so mixed runs are told apart and resume does not reuse them.
- [Truncating `llama` to its first tool call can hide that it planned badly.] → This is acceptable for the thesis because it is applied consistently. It is documented as a per-preset adjustment.
- [Argument decoding can mask a model that wrongly stringifies on purpose.] → Decoding only applies when the decoded value matches the declared type. Everything else still fails.
- [The `devel` image is about 6 GB larger.] → There is plenty of disk on the RTX host (788 GB free).

## Migration Plan

1. Pull the branch and rebuild the image (`make docker-up` builds it).
2. Remove `AGENT_MAX_STEPS=10` from any local `.env` so the new default applies, or set the value explicitly.
3. Start new result files, because old ones have a different config hash.
