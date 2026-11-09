# vLLM Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LM Studio with a vLLM server: Docker Compose + CUDA on NVIDIA (Linux/Windows), native `vllm-metal` on macOS, five shared presets, same OpenAI-compatible API on port 8000.

**Architecture:** Each preset is a `vllm serve --config` YAML file, one per platform (`presets/cuda/`, `presets/metal/`). Compose runs one profile per CUDA preset from an image built from the `llm` uv dependency group; macOS runs the Metal preset natively. The Python client only changes defaults and sends `max_tokens` plus a thinking toggle.

**Tech Stack:** vLLM 0.33.0, `nvidia/cuda:13.0.2-devel-ubuntu24.04`, uv 0.13.7, Docker Compose, vllm-metal (Homebrew), langchain-openai, pydantic-settings, pytest, PyYAML (tests only).

Spec: `docs/superpowers/specs/2026-11-09-vllm-server-design.md`.

## Global Constraints

- Preset names (= `served-model-name` = Compose profile = `LLM_MODEL`): `qwen`, `gemma`, `llama`, `qwen-mid`, `qwen-weak`.
- Server port: `8000`; client default `LLM_BASE_URL=http://localhost:8000/v1`.
- vLLM pinned to `0.33.0`; `llm` group marker: `sys_platform == 'linux' and python_version < '3.15'`.
- Base image: `nvidia/cuda:13.0.2-devel-ubuntu24.04`; uv image `ghcr.io/astral-sh/uv:0.13.7`. Host NVIDIA driver ≥ 580.
- All presets: `max-model-len: 32768`, `enable-auto-tool-choice: true`.
- CUDA budgets (16 GB card): `qwen`/`gemma`/`llama` 0.75, `qwen-mid` 0.375, `qwen-weak` 0.25.
- `llm_max_tokens` default 4096; `llm_enable_thinking` default `false`, sent as `chat_template_kwargs.enable_thinking`.
- Evaluation CLI sampling unchanged (`--temperature` 0.7, 10 repeats).
- Comments: only what code cannot show, one short line. Ruff (line length 100, google docstrings) and `mypy --strict` must pass.

## File Structure

| Path | Action | Responsibility |
| --- | --- | --- |
| `src/tool_eval/config.py` | Modify | New defaults + `llm_max_tokens`, `llm_enable_thinking` |
| `src/tool_eval/llm.py` | Modify | Pass max tokens and thinking flag to `ChatOpenAI` |
| `tests/test_llm.py` | Create | Client defaults and request parameters |
| `src/tool_eval/server/presets/cuda/*.yaml` | Create | 5 CUDA presets |
| `src/tool_eval/server/presets/metal/*.yaml` | Create | 5 Metal presets |
| `src/tool_eval/server/templates/*.jinja` | Create | Gemma 4 / Llama 3.1 tool templates from vLLM v0.33.0 |
| `tests/test_server_presets.py` | Create | Preset and Compose consistency |
| `pyproject.toml`, `uv.lock` | Modify | `llm` group; PyYAML dev deps |
| `src/tool_eval/server/Dockerfile` | Create | CUDA vLLM image |
| `.dockerignore` | Create | Trim root build context |
| `docker-compose.yml` | Create | One profile per CUDA preset |
| `.env.example`, `README.md`, `tests/test_evaluation.py` | Modify | Docs and LM Studio wording |

---

### Task 1: Client settings and chat model

**Files:**

- Modify: `src/tool_eval/config.py:25-27`
- Modify: `src/tool_eval/llm.py`
- Create: `tests/test_llm.py`

**Interfaces:**

- Produces: `Settings.llm_max_tokens: int`, `Settings.llm_enable_thinking: bool`; `build_chat_model(settings)` returns `ChatOpenAI` with `max_tokens` and `extra_body` set.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm.py`:

```python
import pytest

from tool_eval.config import Settings
from tool_eval.llm import build_chat_model


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_MAX_TOKENS", "LLM_ENABLE_THINKING"):
        monkeypatch.delenv(name, raising=False)


def test_defaults_target_local_vllm() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_base_url == "http://localhost:8000/v1"
    assert settings.llm_model == "qwen"
    assert settings.llm_max_tokens == 4096
    assert settings.llm_enable_thinking is False


def test_chat_model_sends_max_tokens_and_thinking_flag() -> None:
    settings = Settings(_env_file=None, llm_max_tokens=512, llm_enable_thinking=True)
    model = build_chat_model(settings)
    assert model.max_tokens == 512
    assert model.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL — `llm_base_url` is `http://localhost:1234/v1`, `Settings` has no `llm_max_tokens`.

- [ ] **Step 3: Implement**

In `src/tool_eval/config.py` replace the three `llm_*` lines with:

```python
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: SecretStr = SecretStr("EMPTY")
    llm_model: str = "qwen"
    llm_max_tokens: int = Field(default=4096, ge=1)
    llm_enable_thinking: bool = False
```

Replace `src/tool_eval/llm.py` body:

```python
"""Construction of the chat model that talks to the vLLM server (OpenAI-compatible API)."""

from langchain_openai import ChatOpenAI

from tool_eval.config import Settings, get_settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """Create a `ChatOpenAI` client pointed at the configured local endpoint.

    Args:
        settings: Application settings.

    Returns:
        The configured chat model.
    """
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.agent_temperature,
        max_completion_tokens=settings.llm_max_tokens,
        # Read by the Qwen3.5 / Gemma 4 chat templates; ignored by templates without thinking.
        extra_body={"chat_template_kwargs": {"enable_thinking": settings.llm_enable_thinking}},
    )
```

Keep `get_chat_model()` unchanged.

- [ ] **Step 4: Run tests and static checks**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy src tests`
Expected: all tests PASS, no ruff or mypy errors.

- [ ] **Step 5: Commit**

```bash
git add src/tool_eval/config.py src/tool_eval/llm.py tests/test_llm.py
git commit -m "feat: point client at vLLM and add max tokens / thinking settings"
```

---

### Task 2: Presets, templates, consistency test

**Files:**

- Modify: `pyproject.toml` (dev group)
- Create: `tests/test_server_presets.py`
- Create: `src/tool_eval/server/presets/cuda/{qwen,gemma,llama,qwen-mid,qwen-weak}.yaml`
- Create: `src/tool_eval/server/presets/metal/{qwen,gemma,llama,qwen-mid,qwen-weak}.yaml`
- Create: `src/tool_eval/server/templates/tool_chat_template_gemma4.jinja`
- Create: `src/tool_eval/server/templates/tool_chat_template_llama3.1_json.jinja`

**Interfaces:**

- Produces: preset files at `src/tool_eval/server/presets/<platform>/<name>.yaml`; CUDA presets reference templates at `/app/templates/<file>`, Metal presets at `src/tool_eval/server/templates/<file>`. Task 4 copies `presets/cuda` to `/app/presets/cuda` and `templates` to `/app/templates`.

- [ ] **Step 1: Add PyYAML to dev dependencies**

Run: `uv add --dev pyyaml types-PyYAML`
Expected: `pyproject.toml` dev group gains both packages; `uv.lock` updated.

- [ ] **Step 2: Write the failing test**

Create `tests/test_server_presets.py`:

```python
from pathlib import Path
from typing import Any

import pytest
import yaml

SERVER_DIR = Path(__file__).resolve().parents[1] / "src" / "tool_eval" / "server"
PRESET_NAMES = {"qwen", "gemma", "llama", "qwen-mid", "qwen-weak"}
SHARED_KEYS = ("served-model-name", "tool-call-parser", "reasoning-parser", "max-model-len")


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), path
    return data


def _presets(platform: str) -> dict[str, dict[str, Any]]:
    return {p.stem: _load(p) for p in (SERVER_DIR / "presets" / platform).glob("*.yaml")}


@pytest.mark.parametrize("platform", ["cuda", "metal"])
def test_platform_defines_all_presets(platform: str) -> None:
    presets = _presets(platform)
    assert set(presets) == PRESET_NAMES
    for name, preset in presets.items():
        assert preset["served-model-name"] == name
        assert preset["enable-auto-tool-choice"] is True
        assert preset["max-model-len"] == 32768


def test_platforms_share_parsers_and_templates() -> None:
    cuda, metal = _presets("cuda"), _presets("metal")
    for name in PRESET_NAMES:
        for key in SHARED_KEYS:
            assert cuda[name].get(key) == metal[name].get(key), (name, key)
        cuda_template = cuda[name].get("chat-template")
        metal_template = metal[name].get("chat-template")
        assert (cuda_template is None) == (metal_template is None), name
        if cuda_template and metal_template:
            assert Path(cuda_template).name == Path(metal_template).name
            assert (SERVER_DIR / "templates" / Path(cuda_template).name).is_file()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_server_presets.py -v`
Expected: FAIL — `set() != {...}` (no preset files yet).

- [ ] **Step 4: Download the chat templates from vLLM v0.33.0**

```bash
mkdir -p src/tool_eval/server/templates
curl -fsSL -o src/tool_eval/server/templates/tool_chat_template_gemma4.jinja \
  https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/examples/tool_chat_template_gemma4.jinja
curl -fsSL -o src/tool_eval/server/templates/tool_chat_template_llama3.1_json.jinja \
  https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/examples/tool_chat_template_llama3.1_json.jinja
```

Expected: both files non-empty; Gemma starts with `{%- macro format_parameters`, Llama with `{{- bos_token }}`.

- [ ] **Step 5: Create CUDA presets**

`src/tool_eval/server/presets/cuda/qwen.yaml`:

```yaml
model: cyankiwi/Qwen3.5-9B-AWQ-4bit
served-model-name: qwen
max-model-len: 32768
gpu-memory-utilization: 0.75
enable-auto-tool-choice: true
tool-call-parser: qwen3_coder
reasoning-parser: qwen3
language-model-only: true
```

`src/tool_eval/server/presets/cuda/gemma.yaml`:

```yaml
model: google/gemma-4-12B-it-qat-w4a16-ct
served-model-name: gemma
max-model-len: 32768
gpu-memory-utilization: 0.75
enable-auto-tool-choice: true
tool-call-parser: gemma4
reasoning-parser: gemma4
chat-template: /app/templates/tool_chat_template_gemma4.jinja
limit-mm-per-prompt: '{"image": 0, "audio": 0}'
```

`src/tool_eval/server/presets/cuda/llama.yaml`:

```yaml
model: hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
served-model-name: llama
max-model-len: 32768
gpu-memory-utilization: 0.75
enable-auto-tool-choice: true
tool-call-parser: llama3_json
chat-template: /app/templates/tool_chat_template_llama3.1_json.jinja
```

`src/tool_eval/server/presets/cuda/qwen-mid.yaml`:

```yaml
model: Qwen/Qwen3.5-4B
served-model-name: qwen-mid
max-model-len: 32768
gpu-memory-utilization: 0.375
quantization: fp8
enable-auto-tool-choice: true
tool-call-parser: qwen3_coder
reasoning-parser: qwen3
language-model-only: true
```

`src/tool_eval/server/presets/cuda/qwen-weak.yaml`:

```yaml
model: Qwen/Qwen3.5-0.8B
served-model-name: qwen-weak
max-model-len: 32768
gpu-memory-utilization: 0.25
enable-auto-tool-choice: true
tool-call-parser: qwen3_coder
reasoning-parser: qwen3
language-model-only: true
```

- [ ] **Step 6: Create Metal presets**

`src/tool_eval/server/presets/metal/qwen.yaml`:

```yaml
model: mlx-community/Qwen3.5-9B-MLX-4bit
served-model-name: qwen
max-model-len: 32768
enable-auto-tool-choice: true
tool-call-parser: qwen3_coder
reasoning-parser: qwen3
```

`src/tool_eval/server/presets/metal/gemma.yaml`:

```yaml
model: mlx-community/gemma-4-12B-it-4bit
served-model-name: gemma
max-model-len: 32768
enable-auto-tool-choice: true
tool-call-parser: gemma4
reasoning-parser: gemma4
chat-template: src/tool_eval/server/templates/tool_chat_template_gemma4.jinja
```

`src/tool_eval/server/presets/metal/llama.yaml`:

```yaml
model: mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
served-model-name: llama
max-model-len: 32768
enable-auto-tool-choice: true
tool-call-parser: llama3_json
chat-template: src/tool_eval/server/templates/tool_chat_template_llama3.1_json.jinja
```

`src/tool_eval/server/presets/metal/qwen-mid.yaml`:

```yaml
model: mlx-community/Qwen3.5-4B-MLX-4bit
served-model-name: qwen-mid
max-model-len: 32768
enable-auto-tool-choice: true
tool-call-parser: qwen3_coder
reasoning-parser: qwen3
```

`src/tool_eval/server/presets/metal/qwen-weak.yaml`:

```yaml
model: mlx-community/Qwen3.5-0.8B-MLX-bf16
served-model-name: qwen-weak
max-model-len: 32768
enable-auto-tool-choice: true
tool-call-parser: qwen3_coder
reasoning-parser: qwen3
```

- [ ] **Step 7: Run tests and static checks**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy src tests`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock tests/test_server_presets.py src/tool_eval/server
git commit -m "feat: add vLLM presets for CUDA and Metal with tool-call templates"
```

---

### Task 3: `llm` dependency group

**Files:**

- Modify: `pyproject.toml` (`[dependency-groups]`)
- Modify: `uv.lock`

**Interfaces:**

- Produces: `uv sync --frozen --only-group llm --no-install-project` installs `vllm==0.33.0` on Linux (used by Task 4).

- [ ] **Step 1: Add the group**

In `pyproject.toml` under `[dependency-groups]`, after `dev = [...]`, add:

```toml
llm = [
    "vllm==0.33.0; sys_platform == 'linux' and python_version < '3.15'",
]
```

- [ ] **Step 2: Lock and verify on macOS**

Run: `uv lock && uv sync && uv run pytest -q`
Expected: lock succeeds; `uv sync` (default groups) installs no vLLM/torch on macOS; tests PASS.

- [ ] **Step 3: Verify the Linux resolution**

Run: `uv tree --only-group llm --depth 1 --python-platform x86_64-unknown-linux-gnu --python-version 3.12`
Expected: `vllm v0.33.0` with `torch v2.14.0` among its children.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add Linux-only llm dependency group with vLLM 0.33.0"
```

---

### Task 4: Dockerfile, .dockerignore, Compose

**Files:**

- Create: `src/tool_eval/server/Dockerfile`
- Create: `.dockerignore`
- Create: `docker-compose.yml`
- Modify: `tests/test_server_presets.py` (add Compose test)

**Interfaces:**

- Consumes: `llm` group (Task 3); `presets/cuda/*.yaml` and `templates/*.jinja` (Task 2).
- Produces: services/profiles `qwen`, `gemma`, `llama`, `qwen-mid`, `qwen-weak` on `localhost:8000`.

- [ ] **Step 1: Write the failing Compose test**

Append to `tests/test_server_presets.py`:

```python
def test_compose_runs_each_cuda_preset() -> None:
    compose = _load(SERVER_DIR.parents[2] / "docker-compose.yml")
    services = compose["services"]
    assert set(services) == PRESET_NAMES
    for name, service in services.items():
        assert service["profiles"] == [name]
        assert service["command"] == ["--config", f"/app/presets/cuda/{name}.yaml"]
```

Run: `uv run pytest tests/test_server_presets.py::test_compose_runs_each_cuda_preset -v`
Expected: FAIL — `FileNotFoundError` for `docker-compose.yml`.

- [ ] **Step 2: Create the Dockerfile**

`src/tool_eval/server/Dockerfile`:

```dockerfile
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group llm --no-install-project
ENV PATH=/app/.venv/bin:$PATH
# Precompiled FlashInfer kernels; otherwise Blackwell JIT-compiles them on every cold start.
RUN vllm download-kernels

COPY src/tool_eval/server/templates /app/templates
COPY src/tool_eval/server/presets/cuda /app/presets/cuda

EXPOSE 8000
ENTRYPOINT ["vllm", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Create `.dockerignore`**

```gitignore
.git
.venv
.env
workspace
results
logs
docs
tests
**/__pycache__
.mypy_cache
.pytest_cache
.ruff_cache
```

- [ ] **Step 4: Create `docker-compose.yml`**

```yaml
x-vllm-base: &vllm-base
  build:
    context: .
    dockerfile: src/tool_eval/server/Dockerfile
  image: tool-eval-vllm:0.33.0
  ports:
    - "8000:8000"
  ipc: host
  environment:
    HF_TOKEN: ${HF_TOKEN:-}
  volumes:
    - ${HF_CACHE_DIR:-~/.cache/huggingface}:/root/.cache/huggingface
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  healthcheck:
    test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
    interval: 15s
    timeout: 5s
    retries: 5
    # First start downloads weights.
    start_period: 30m

services:
  qwen:
    <<: *vllm-base
    profiles: [qwen]
    command: ["--config", "/app/presets/cuda/qwen.yaml"]
  gemma:
    <<: *vllm-base
    profiles: [gemma]
    command: ["--config", "/app/presets/cuda/gemma.yaml"]
  llama:
    <<: *vllm-base
    profiles: [llama]
    command: ["--config", "/app/presets/cuda/llama.yaml"]
  qwen-mid:
    <<: *vllm-base
    profiles: [qwen-mid]
    command: ["--config", "/app/presets/cuda/qwen-mid.yaml"]
  qwen-weak:
    <<: *vllm-base
    profiles: [qwen-weak]
    command: ["--config", "/app/presets/cuda/qwen-weak.yaml"]
```

- [ ] **Step 5: Run tests and validate Compose**

Run: `uv run pytest -q && docker compose --profile qwen config --quiet && docker compose --profile qwen config | grep -A1 "source:"`
Expected: tests PASS; `config --quiet` exits 0; the bind source is an absolute path ending in `/.cache/huggingface` (`~` expanded).

- [ ] **Step 6: Commit**

```bash
git add src/tool_eval/server/Dockerfile .dockerignore docker-compose.yml tests/test_server_presets.py
git commit -m "feat: add CUDA vLLM image and compose profiles per preset"
```

---

### Task 5: Docs, env example, LM Studio wording

**Files:**

- Modify: `.env.example:1-3`
- Modify: `README.md:12`
- Modify: `tests/test_evaluation.py:331-332`

- [ ] **Step 1: Update `.env.example`**

Replace the first three lines with:

```dotenv
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=EMPTY
# Served name of the running preset: qwen | gemma | llama | qwen-mid | qwen-weak
LLM_MODEL=qwen
LLM_MAX_TOKENS=4096
LLM_ENABLE_THINKING=false

# Hugging Face token for gated models; HF_CACHE_DIR overrides the Docker weight cache mount
HF_TOKEN=
# HF_CACHE_DIR=
```

- [ ] **Step 2: Update `README.md`**

Replace the line `Start the LM Studio local server ...` with:

````markdown
## Model server

Presets (set `LLM_MODEL` in `.env` to the preset name): `qwen`, `gemma`, `llama` (≤ 12 GB VRAM), `qwen-mid` (~6 GB), `qwen-weak` (~4 GB). Definitions: `src/tool_eval/server/presets/{cuda,metal}/`. Both paths serve `http://localhost:8000/v1`.

NVIDIA GPU — Linux, or Windows with Docker Desktop (WSL 2 backend); driver ≥ 580:

```sh
docker compose --profile qwen up -d --build
```

macOS (Apple Silicon, GPU via vllm-metal; Docker on macOS has no GPU access):

```sh
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
vllm serve --config src/tool_eval/server/presets/metal/qwen.yaml   # run from the repo root
```

Gated models need `HF_TOKEN` (in `.env` for Docker, exported in the shell for macOS). Mac presets use MLX 4-bit weights, so their results are not directly comparable with the CUDA runs.
````

- [ ] **Step 3: Rename the test message**

In `tests/test_evaluation.py` rename `lm_down` → `server_down` (both occurrences) and change the message to `"vLLM server unreachable"`.

- [ ] **Step 4: Run checks**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy src tests && grep -rn "LM Studio\|1234" --exclude-dir=.venv --exclude-dir=workspace --exclude-dir=docs --include="*.py" --include="*.md" --include=".env.example" .`
Expected: tests PASS, no lint/type errors, grep prints nothing.

- [ ] **Step 5: Commit**

```bash
git add .env.example README.md tests/test_evaluation.py
git commit -m "docs: document vLLM model server for NVIDIA and macOS"
```

---

### Task 6: Manual verification

No code changes unless a check fails. Record results in the PR description.

- [ ] **Step 1: macOS — weak preset end to end**

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
vllm serve --config src/tool_eval/server/presets/metal/qwen-weak.yaml
```

In a second terminal:

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool | grep '"id"'
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen-weak",
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "chat_template_kwargs": {"enable_thinking": false}
}' | python3 -m json.tool
```

Expected: `"id": "qwen-weak"`; response has `choices[0].message.tool_calls[0].function.name == "get_weather"`.

Then with `LLM_MODEL=qwen-weak` in `.env`:

Run: `uv run python -m tool_eval.main run --only rag-2024h1 --levels level1 --repeats 1`
Expected: one record written to `results/results.jsonl` with no connection error.

- [ ] **Step 2: macOS — remaining presets start**

For each of `qwen`, `gemma`, `llama`, `qwen-mid`: start `vllm serve --config src/tool_eval/server/presets/metal/<name>.yaml`, repeat the tool-call `curl` with `"model": "<name>"`, stop with Ctrl-C.
Expected: each returns a structured `tool_calls`. If `--config` rejects `model`, move it to the command (`vllm serve <model> --config ...`) in README and Compose `command`.

- [ ] **Step 3: RTX 5080 — build and serve each profile**

```bash
docker compose --profile qwen up -d --build
docker compose --profile qwen ps   # wait for "healthy"
```

Repeat the `/v1/models` and tool-call `curl` from Step 1 with `"model": "qwen"`, then `docker compose --profile qwen down`. Repeat for `gemma`, `llama`, `qwen-mid`, `qwen-weak`.
Expected: each container becomes `healthy` and returns structured `tool_calls`. If `vllm download-kernels` fails during the build because no GPU is visible, delete that `RUN` line (kernels are then fetched at first start). If `qwen-mid` runs out of memory, switch its `model` to `sanskar003/Qwen3.5-4B-AWQ` and remove `quantization: fp8`.

- [ ] **Step 4: Llama parallel tool calls**

With the `llama` profile running, set `LLM_MODEL=llama` and run: `uv run python -m tool_eval.main run --only rag-2024h1 --levels level2 --repeats 1`
Expected: no `This model only supports single tool-calls at once!` error in the record. If it appears, report it — fixing it (e.g. `parallel_tool_calls=False` when binding tools) is a separate change.
