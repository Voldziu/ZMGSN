import pytest
from pydantic import ValidationError

from tool_eval.config import Settings
from tool_eval.llm import build_chat_model


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_MAX_TOKENS",
        "LLM_ENABLE_THINKING",
        "LLM_PARALLEL_TOOL_CALLS",
        "LLM_TOP_P",
        "LLM_TOP_K",
        "AGENT_TEMPERATURE",
        "AGENT_MAX_STEPS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_target_local_vllm() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_base_url == "http://localhost:8000/v1"
    assert settings.llm_model == "qwen"
    assert settings.llm_max_tokens == 8192
    assert settings.llm_enable_thinking is False


def test_chat_model_sends_max_tokens_and_thinking_flag() -> None:
    settings = Settings(_env_file=None, llm_max_tokens=512, llm_enable_thinking=True)
    model = build_chat_model(settings)
    assert model.max_tokens == 512
    assert model.extra_body is not None
    assert model.extra_body["chat_template_kwargs"] == {"enable_thinking": True}


def test_default_step_limit_is_100() -> None:
    assert Settings(_env_file=None).agent_max_steps == 100


def test_loop_guard_defaults_and_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)
    assert settings.agent_max_repeated_calls == 8
    assert settings.agent_max_tool_error_streak == 10
    monkeypatch.setenv("AGENT_MAX_REPEATED_CALLS", "0")
    assert Settings(_env_file=None).agent_max_repeated_calls == 0
    with pytest.raises(ValidationError):
        Settings(_env_file=None, agent_max_tool_error_streak=-1)


@pytest.mark.parametrize(
    ("model", "override", "expected"),
    [
        ("llama", None, False),
        ("qwen", None, False),
        ("gemma", None, False),
        ("qwen", True, True),
        ("llama", False, False),
    ],
)
def test_parallel_tool_call_policy(model: str, override: bool | None, expected: bool) -> None:
    settings = Settings(_env_file=None, llm_model=model, llm_parallel_tool_calls=override)
    assert settings.effective_parallel_tool_calls == expected


@pytest.mark.parametrize("model", ["qwen", "gemma", "llama", "qwen-mid", "qwen-weak"])
def test_every_preset_sends_one_call_per_reply_and_the_same_sampling(model: str) -> None:
    chat = build_chat_model(Settings(_env_file=None, llm_model=model))
    body = chat.extra_body
    assert body is not None
    assert body["parallel_tool_calls"] is False
    assert body["top_k"] == 20
    assert chat.top_p == 0.95
    assert chat.temperature == 0.7


def test_sampling_follows_settings() -> None:
    chat = build_chat_model(
        Settings(_env_file=None, llm_top_p=0.8, llm_top_k=40, agent_temperature=0.2)
    )
    assert chat.extra_body is not None
    assert chat.extra_body["top_k"] == 40
    assert chat.top_p == 0.8
    assert chat.temperature == 0.2


def test_seed_is_sent_when_set() -> None:
    assert build_chat_model(Settings(_env_file=None, llm_seed=1234)).seed == 1234


def test_no_seed_by_default() -> None:
    assert build_chat_model(Settings(_env_file=None)).seed is None
