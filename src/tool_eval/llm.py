"""Construction of the chat model that talks to the vLLM server (OpenAI-compatible API)."""

from typing import Any

from langchain_openai import ChatOpenAI

from tool_eval.config import Settings, get_settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """Create a `ChatOpenAI` client pointed at the configured local endpoint.

    Args:
        settings: Application settings.

    Returns:
        The configured chat model.
    """
    # Read by the Qwen3.5 / Gemma 4 chat templates; ignored by templates without thinking.
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": settings.llm_enable_thinking},
        # vLLM keeps only the first tool call of a reply when this is false.
        "parallel_tool_calls": settings.effective_parallel_tool_calls,
        # Not part of the OpenAI API; vLLM reads it from the request body.
        "top_k": settings.llm_top_k,
    }
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.agent_temperature,
        top_p=settings.llm_top_p,
        max_completion_tokens=settings.llm_max_tokens,
        # vLLM otherwise samples deterministically from server start, so repeat n of a session
        # would reproduce exactly after every restart.
        seed=settings.llm_seed,
        extra_body=extra_body,
    )


def get_chat_model() -> ChatOpenAI:
    """Build a chat model from the active settings (honours `use_settings` overrides).

    Returns:
        The chat model.
    """
    return build_chat_model(get_settings())
