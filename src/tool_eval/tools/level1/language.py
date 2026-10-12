"""Language tools backed by the LLM: summarize, extract keywords, translate."""

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from tool_eval.config import get_settings
from tool_eval.llm import get_chat_model
from tool_eval.services import papers


@tool(parse_docstring=True)
def summarize_chunk(chunk_path: str) -> str:
    """Summarize a text file (typically one chunk of a paper) in a few sentences.

    Args:
        chunk_path: Workspace-relative path of the text file.

    Returns:
        The summary.
    """
    return papers.summarize_chunk_file(chunk_path, settings=get_settings(), llm=get_chat_model())


@tool(parse_docstring=True)
def extract_keywords(text_path: str, max_keywords: Annotated[int, Field(ge=1, le=15)] = 5) -> str:
    """Extract the main keywords from a text file.

    Args:
        text_path: Workspace-relative path of the text file.
        max_keywords: Maximum number of keywords (1-15).

    Returns:
        JSON list of keywords.
    """
    keywords = papers.keywords_from_file(
        text_path,
        max_keywords=max_keywords,
        settings=get_settings(),
        llm=get_chat_model(),
    )
    return json.dumps(keywords, ensure_ascii=False)


@tool(parse_docstring=True)
def translate_text(text_path: str, target_language: str) -> str:
    """Translate a text file into another language and save the result.

    Args:
        text_path: Workspace-relative path of the text file.
        target_language: Language name, for example Polish.

    Returns:
        Workspace-relative path of the translated file.
    """
    return papers.translate_file(
        text_path,
        target_language=target_language,
        settings=get_settings(),
        llm=get_chat_model(),
    )
