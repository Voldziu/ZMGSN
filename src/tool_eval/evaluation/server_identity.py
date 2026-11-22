"""Identity of the served model: checkpoint, quantisation, parser, chat template, vLLM version.

The runner reads it once per evaluation and puts it into the configuration hash, the run
records and the MLflow parameters, so runs against differently served models never look
comparable. Sources:

- the server's `GET /v1/models` (served checkpoint) and `GET /version` (vLLM version);
- the preset file named by `TOOL_EVAL_PRESET_FILE` (set by `eval-presets.sh`): its hash and its
  `quantization`, `tool-call-parser` and `chat-template` entries;
- the chat template the preset names, looked up in `server/templates/`.

A source that cannot be read yields `unknown`; the hash still differs from a known identity.
"""

import hashlib
import os
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

PRESET_FILE_VARIABLE = "TOOL_EVAL_PRESET_FILE"
UNKNOWN = "unknown"
_TEMPLATES = Path(__file__).resolve().parent.parent / "server" / "templates"


class ServerIdentity(BaseModel):
    """What the client knows about the model server behind `LLM_BASE_URL`."""

    model_config = ConfigDict(frozen=True)

    checkpoint: str = UNKNOWN
    vllm_version: str = UNKNOWN
    quantization: str = UNKNOWN
    tool_parser: str = UNKNOWN
    chat_template_sha256: str = UNKNOWN
    preset_sha256: str = UNKNOWN


def read_server_identity(
    base_url: str,
    *,
    served_name: str,
    preset_file: Path | None = None,
    timeout: float = 5.0,
) -> ServerIdentity:
    """Collect the identity of the served model, falling back to `unknown` per source.

    Args:
        base_url: OpenAI-compatible base URL, for example `http://localhost:8000/v1`.
        served_name: Served model name (`LLM_MODEL`).
        preset_file: Preset YAML; defaults to `TOOL_EVAL_PRESET_FILE` when set.
        timeout: HTTP timeout in seconds.

    Returns:
        The identity.
    """
    if preset_file is None and os.environ.get(PRESET_FILE_VARIABLE):
        preset_file = Path(os.environ[PRESET_FILE_VARIABLE])
    preset = _read_preset(preset_file)
    return ServerIdentity(
        checkpoint=_served_checkpoint(base_url, served_name, timeout=timeout),
        vllm_version=_vllm_version(base_url, timeout=timeout),
        quantization=preset.get("quantization", "none") if preset else UNKNOWN,
        tool_parser=preset.get("tool-call-parser", UNKNOWN) if preset else UNKNOWN,
        chat_template_sha256=_template_hash(preset),
        preset_sha256=_sha256(preset_file.read_bytes()) if preset and preset_file else UNKNOWN,
    )


def _read_preset(path: Path | None) -> dict[str, str] | None:
    """Parse the flat `key: value` lines of a preset file, or `None` when unreadable."""
    if path is None:
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    entries: dict[str, str] = {}
    for line in lines:
        content = line.split("#", 1)[0].strip()
        if ":" in content:
            key, value = content.split(":", 1)
            entries[key.strip()] = value.strip()
    return entries


def _template_hash(preset: dict[str, str] | None) -> str:
    if preset is None:
        return UNKNOWN
    template = preset.get("chat-template")
    if not template:
        return "model-default"
    try:
        return _sha256((_TEMPLATES / Path(template).name).read_bytes())
    except OSError:
        return UNKNOWN


def _served_checkpoint(base_url: str, served_name: str, *, timeout: float) -> str:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        response.raise_for_status()
        for entry in response.json().get("data", []):
            if entry.get("id") == served_name:
                return str(entry.get("root") or served_name)
    except (httpx.HTTPError, ValueError, AttributeError):
        return UNKNOWN
    return UNKNOWN


def _vllm_version(base_url: str, *, timeout: float) -> str:
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        response = httpx.get(f"{root}/version", timeout=timeout)
        response.raise_for_status()
        return str(response.json().get("version", UNKNOWN))
    except (httpx.HTTPError, ValueError, AttributeError):
        return UNKNOWN


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
