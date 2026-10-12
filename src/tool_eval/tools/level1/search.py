"""Search tools: arXiv and Wikipedia."""

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from tool_eval.config import get_settings
from tool_eval.services import arxiv, wikipedia


@tool(parse_docstring=True)
def search_arxiv(query: str, max_results: Annotated[int, Field(ge=1, le=10)] = 5) -> str:
    """Search arXiv for recent papers on a topic, newest first.

    Args:
        query: Keywords describing the topic.
        max_results: Number of papers to return (1-10).

    Returns:
        JSON list of papers with arxiv_id, title, authors, published date and abstract.
    """
    settings = get_settings()
    papers = arxiv.search_arxiv(
        query,
        max_results=max_results,
        date_from=settings.arxiv_date_from,
        date_to=settings.arxiv_date_to,
        timeout=settings.http_timeout_seconds,
    )
    return json.dumps([paper.model_dump(mode="json") for paper in papers], ensure_ascii=False)


@tool(parse_docstring=True)
def search_wikipedia(query: str, max_results: Annotated[int, Field(ge=1, le=5)] = 3) -> str:
    """Search Wikipedia for background information on a topic.

    Args:
        query: Search terms.
        max_results: Number of articles to return (1-5).

    Returns:
        JSON list of articles with title, extract and url.
    """
    settings = get_settings()
    results = wikipedia.search_wikipedia(
        query,
        max_results=max_results,
        timeout=settings.http_timeout_seconds,
    )
    return json.dumps([item.model_dump(mode="json") for item in results], ensure_ascii=False)
