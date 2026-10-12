"""Construction of the chat model that talks to LM Studio (OpenAI-compatible API)."""

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
    )


def get_chat_model() -> ChatOpenAI:
    """Build a chat model from the active settings (honours `use_settings` overrides).

    Returns:
        The chat model.
    """
    return build_chat_model(get_settings())
