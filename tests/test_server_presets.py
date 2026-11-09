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
            assert cuda_template.startswith("/app/templates/"), name
            assert (SERVER_DIR.parents[2] / metal_template).is_file(), name
            assert Path(cuda_template).name == Path(metal_template).name
            assert (SERVER_DIR / "templates" / Path(cuda_template).name).is_file()


def test_compose_runs_each_cuda_preset() -> None:
    compose = _load(SERVER_DIR.parents[2] / "docker-compose.yml")
    services = compose["services"]
    assert set(services) == PRESET_NAMES
    for name, service in services.items():
        assert service["profiles"] == [name]
        assert service["command"] == ["--config", f"/app/presets/cuda/{name}.yaml"]
