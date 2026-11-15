# Tasks

## 1. CUDA server (already applied on the branch)

- [x] 1.1 Pin `nvidia-cuda-nvcc`, `nvidia-cuda-crt` and `nvidia-nvvm` to 13.0.88 in the `llm` group (`92c4de3`). Verified: `nvcc --version` in the image reports 13.0, matching `CUDART_VERSION 13000`.
- [x] 1.2 Build the server image on `nvidia/cuda:13.0.2-devel-ubuntu24.04` with `CUDA_HOME=/usr/local/cuda` (`fd1ce13`). Verified: FlashInfer JIT kernels compile and link on the RTX 5080.
- [x] 1.3 Set `max-num-seqs: 8` in every CUDA preset (`20f3aaf`). Verified by `test_cuda_presets_limit_concurrent_sequences` and `qwen-weak` starting at 4 GB.
- [x] 1.4 Raise the `qwen-mid` CUDA budget to 0.5 and update README and the design notes (`250d181`). Verified: `qwen-mid` starts in about 180 s and passes `tool-check`.
- [x] 1.5 Build the `tool-check` request in a make variable (`60b2f1d`). Verified: valid JSON with GNU Make 4.3 and macOS make 3.81.
- [x] 1.6 Document the NVIDIA host prerequisites (driver, Container Toolkit with `nvidia-ctk runtime configure`, docker group) and the GPU check command in README. Verify that the README commands match what was run on the host.

## 2. Client: parallel tool calls

- [x] 2.1 Add the `llm_parallel_tool_calls: bool | None` setting and a preset default (`false` for `llama`). Send `parallel_tool_calls: false` in the request body when the effective value is false. Verify with unit tests for `llama`, `qwen` and an explicit override.
- [x] 2.2 Add the effective parallel-call policy to `config_fingerprint`. Verify with a test that the hash changes when only the policy changes.
- [x] 2.3 Document `LLM_PARALLEL_TOOL_CALLS` in `.env.example` and README. Verify that the documented values match the setting.

## 3. Agent: argument decoding and step limit

- [x] 3.1 Add a tool-call middleware that decodes JSON-string values for `array`/`object` arguments before validation. Verify with tests: a stringified list is accepted, a non-JSON string still fails, and a string argument stays unchanged.
- [x] 3.2 Change the default `agent_max_steps` from 10 to 30 and update `.env.example` and the docs. Verify with a settings-default test and a test that `max_steps` changes the config hash.

## 4. Multi-preset evaluation command

- [ ] 4.1 Add `make eval-presets` (`PRESETS`, `REPEATS`, `ONLY`, `LEVELS`) backed by `src/tool_eval/server/eval-presets.sh`. It starts vllm-metal on macOS or Docker Compose elsewhere, waits for readiness, runs the evaluation into `results/presets-<preset>.jsonl`, stops the server (also on Ctrl-C), and prints the per-level report. Verify with an unknown preset (rejected), and with a real run of `qwen-weak` on the Mac and on the RTX.
- [x] 4.2 Document `make eval-presets` in README and verify that `make` lists it.

## 5. Verification on hardware

- [ ] 5.1 On the RTX 5080, run each of the 5 presets with 3 repeats at the default limit. Verify that no run ends with HTTP 400, that `llama` and `gemma` can complete, and record status, score, calls and tool errors per preset.
- [x] 5.2 Run the unit tests, `ruff` and `mypy`, and verify they all pass.

## Workflow follow-up

- Archive the change once section 5 is done, which syncs the three specs.
