from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from tool_eval.evaluation import server_identity
from tool_eval.evaluation.server_identity import UNKNOWN, read_server_identity


def _fake_get(
    models: dict[str, object] | None, version: str | None
) -> Callable[..., httpx.Response]:
    def get(url: str, **_: object) -> httpx.Response:
        request = httpx.Request("GET", url)
        if url.endswith("/models") and models is not None:
            return httpx.Response(200, json=models, request=request)
        if url.endswith("/version") and version is not None:
            return httpx.Response(200, json={"version": version}, request=request)
        raise httpx.ConnectError("refused", request=request)

    return get


def test_identity_from_server_and_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx,
        "get",
        _fake_get({"data": [{"id": "llama", "root": "hugging-quants/Llama-AWQ"}]}, "0.11.0"),
    )
    preset = tmp_path / "llama.yaml"
    preset.write_text(
        "model: x\nserved-model-name: llama\nquantization: awq  # comment\n"
        "tool-call-parser: llama3_json\n"
        "chat-template: /app/templates/tool_chat_template_llama3.1_json.jinja\n",
        encoding="utf-8",
    )
    identity = read_server_identity(
        "http://localhost:8000/v1", served_name="llama", preset_file=preset
    )
    assert identity.checkpoint == "hugging-quants/Llama-AWQ"
    assert identity.vllm_version == "0.11.0"
    assert identity.quantization == "awq"
    assert identity.tool_parser == "llama3_json"
    assert len(identity.chat_template_sha256) == 64
    assert len(identity.preset_sha256) == 64


def test_preset_without_quantization_or_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(httpx, "get", _fake_get(None, None))
    preset = tmp_path / "qwen-mid.yaml"
    preset.write_text("model: Qwen/Qwen3.5-4B\ntool-call-parser: qwen3_coder\n", encoding="utf-8")
    identity = read_server_identity("http://h/v1", served_name="qwen-mid", preset_file=preset)
    assert identity.quantization == "none"
    assert identity.chat_template_sha256 == "model-default"
    assert identity.checkpoint == UNKNOWN
    assert identity.vllm_version == UNKNOWN


def test_nothing_readable_gives_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(server_identity.PRESET_FILE_VARIABLE, raising=False)
    monkeypatch.setattr(httpx, "get", _fake_get(None, None))
    identity = read_server_identity("http://h/v1", served_name="qwen")
    assert set(identity.model_dump().values()) == {UNKNOWN}


def test_preset_file_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", _fake_get(None, None))
    preset = tmp_path / "p.yaml"
    preset.write_text("quantization: fp8\n", encoding="utf-8")
    monkeypatch.setenv(server_identity.PRESET_FILE_VARIABLE, str(preset))
    assert read_server_identity("http://h/v1", served_name="q").quantization == "fp8"
