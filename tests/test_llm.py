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
