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
    assert model.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_default_step_limit_is_100() -> None:
    assert Settings(_env_file=None).agent_max_steps == 100


def test_loop_guard_defaults_and_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)
    assert settings.agent_max_repeated_calls == 5
    assert settings.agent_max_tool_error_streak == 10
    monkeypatch.setenv("AGENT_MAX_REPEATED_CALLS", "0")
    assert Settings(_env_file=None).agent_max_repeated_calls == 0
    with pytest.raises(ValidationError):
        Settings(_env_file=None, agent_max_tool_error_streak=-1)


@pytest.mark.parametrize(
    ("model", "override", "expected"),
    [
        ("llama", None, False),
        ("qwen", None, None),
        ("gemma", None, None),
        ("qwen", False, False),
        ("llama", True, True),
    ],
)
def test_parallel_tool_call_policy(
    model: str, override: bool | None, expected: bool | None
) -> None:
    settings = Settings(_env_file=None, llm_model=model, llm_parallel_tool_calls=override)
    assert settings.effective_parallel_tool_calls == expected


def test_single_tool_call_preset_sends_flag() -> None:
    body = build_chat_model(Settings(_env_file=None, llm_model="llama")).extra_body
    assert body is not None
    assert body["parallel_tool_calls"] is False


def test_other_presets_leave_parallel_calls_to_server() -> None:
    body = build_chat_model(Settings(_env_file=None, llm_model="qwen")).extra_body
    assert body is not None
    assert "parallel_tool_calls" not in body


def test_seed_is_sent_when_set() -> None:
    assert build_chat_model(Settings(_env_file=None, llm_seed=1234)).seed == 1234


def test_no_seed_by_default() -> None:
    assert build_chat_model(Settings(_env_file=None)).seed is None
