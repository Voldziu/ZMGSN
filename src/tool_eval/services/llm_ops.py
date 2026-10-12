"""Single-shot LLM operations used by tools: summarize, extract keywords, translate."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from tool_eval.prompts import Prompt
from tool_eval.utils.text import truncate


def summarize(text: str, *, llm: BaseChatModel, prompt: Prompt, max_chars: int) -> str:
    """Summarize a passage in a few sentences.

    Args:
        text: Passage to summarize; truncated to `max_chars`.
        llm: Chat model.
        prompt: Prompt without placeholders.
        max_chars: Maximum number of input characters sent to the model.

    Returns:
        The summary.
    """
    return _complete(llm, prompt.text, truncate(text, max_chars=max_chars))


def extract_keywords(
    text: str,
    *,
    llm: BaseChatModel,
    prompt: Prompt,
    max_keywords: int,
    max_chars: int,
) -> list[str]:
    """Extract keywords from a text.

    Args:
        text: Text to analyze; truncated to `max_chars`.
        llm: Chat model.
        prompt: Prompt with a `$max_keywords` placeholder.
        max_keywords: Maximum number of keywords to return.
        max_chars: Maximum number of input characters sent to the model.

    Returns:
        Unique keywords in the order produced by the model.
    """
    system = prompt.render(max_keywords=str(max_keywords))
    reply = _complete(llm, system, truncate(text, max_chars=max_chars))
    keywords = [item.strip(" -*\t") for item in reply.replace("\n", ",").split(",")]
    return list(dict.fromkeys(item for item in keywords if item))[:max_keywords]


def translate(
    text: str,
    *,
    llm: BaseChatModel,
    prompt: Prompt,
    target_language: str,
    max_chars: int,
) -> str:
    """Translate a text into another language.

    Args:
        text: Text to translate; truncated to `max_chars`.
        llm: Chat model.
        prompt: Prompt with a `$target_language` placeholder.
        target_language: Language name, for example `Polish`.
        max_chars: Maximum number of input characters sent to the model.

    Returns:
        The translation.
    """
    system = prompt.render(target_language=target_language)
    return _complete(llm, system, truncate(text, max_chars=max_chars))


def _complete(llm: BaseChatModel, system: str, text: str) -> str:
    response = llm.invoke([SystemMessage(system), HumanMessage(text)])
    return response.text.strip()
