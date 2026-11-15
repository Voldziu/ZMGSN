# Tasks

Reconstructed from branch `mikolaj/feat/vllm-instead-of-lmstudio` (commits `c4a9334`..`401e012`). Checked items are implemented on the branch; unchecked items are still open.

## 1. Dependencies

- [x] 1.1 Add the Linux-only `llm` uv group (`vllm==0.33.0`), conflicting with `dev`; add `pyyaml`/`types-pyyaml` to `dev`. Verify that `uv lock` and `uv sync` succeed on macOS.

## 2. Client configuration

- [x] 2.1 Change the `Settings` defaults to `http://localhost:8000/v1`, `EMPTY` and `qwen`, and add `llm_max_tokens` (default 4096, minimum 1) and `llm_enable_thinking` (default false). Verify with `tests/test_llm.py::test_defaults_target_local_vllm`.
- [x] 2.2 Pass `max_completion_tokens` and `extra_body.chat_template_kwargs.enable_thinking` in `build_chat_model`. Verify with `tests/test_llm.py::test_chat_model_sends_max_tokens_and_thinking_flag`.
- [x] 2.3 Rename the LM Studio error in the evaluation test to "vLLM server unreachable". Verify that `tests/test_evaluation.py` passes.

## 3. Presets and templates

- [x] 3.1 Add the five CUDA and five Metal preset YAMLs, plus the Gemma 4 and Llama 3.1 tool chat templates. Verify with `tests/test_server_presets.py::test_platform_defines_all_presets` and `::test_platforms_share_parsers_and_templates`.
- [x] 3.2 Confirm that all ten checkpoints exist on Hugging Face and are not gated. Verify with the HF API (checked 2026-11-15; weights range from 1.7 to 10.3 GB).

## 4. NVIDIA server (Docker)

- [x] 4.1 Add the CUDA 13 base image Dockerfile (toolkit from pip, `g++` from apt, no build-time kernel download) and `.dockerignore`. Verify that the image builds at about 10.2 GB.
- [x] 4.2 Add `docker-compose.yml` with one profile per preset, GPU reservation, HF cache mount and health check. Verify that `docker compose --profile <p> config -q` passes for all five profiles and that `test_compose_runs_each_cuda_preset` passes.

## 5. macOS server and Makefile

- [x] 5.1 Add the `Makefile` targets: `mac-install` (with `install.sh` fallback), `mac-serve`, `docker-up/status/logs/down` (with `docker-up` refused on macOS), and `models`, `tool-check`, `eval-smoke`, `test`. Verify that `make` prints the cheat sheet and that `make PRESET=bogus` fails with the list of presets.
- [x] 5.2 Document the model server in README.md and `.env.example`, and remove the LM Studio wording. Verify that `grep -rn "LM Studio\|1234"` outside `docs/` finds nothing.

## 6. Manual verification

- [ ] 6.1 On the 24 GB Mac, run `make mac-install`, then `make mac-serve PRESET=qwen-weak`, then `make models` and `make tool-check`. Verify that the model id is `qwen-weak` and that `tool_calls` contains `get_weather`.
- [ ] 6.2 On the Mac, run `make eval-smoke PRESET=qwen-weak`. Verify that `results/smoke-qwen-weak.jsonl` is written with one record.
- [ ] 6.3 On the Mac, repeat 6.1 for `qwen`, `gemma`, `llama` and `qwen-mid`. Verify that each serves and returns `tool_calls`, and note memory headroom for `gemma` and `qwen`.
- [ ] 6.4 On the RTX 5080, run `make docker-up`, then `make docker-status`, then `make tool-check` for each of the five presets. Verify that each becomes `healthy` and returns `tool_calls`; if `qwen-mid` does not fit, switch it to `sanskar003/Qwen3.5-4B-AWQ`.
- [ ] 6.5 With the `llama` preset, run an evaluation. Verify whether the agent emits more than one tool call per message, which the Llama 3.1 template rejects.

## Workflow follow-up

- The change was archived at the user's request before section 6 was done, to set up baseline specs. Track the open verification items outside the archived change.
