"""Level 2 tool: search arXiv and get APA references in one call."""

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from tool_eval.config import get_settings
from tool_eval.services import reports


@tool(parse_docstring=True)
def search_papers(query: str, max_results: Annotated[int, Field(ge=1, le=10)] = 5) -> str:
    """Search arXiv for recent papers on a topic and return each with its APA reference.

    Args:
        query: Keywords describing the topic.
        max_results: Number of papers to return (1-10).

    Returns:
        JSON list of papers with arxiv_id, title, published date and apa_citation.
    """
    papers = reports.search_cited_papers(
        query,
        max_results=max_results,
        settings=get_settings(),
    )
    return json.dumps([paper.model_dump(mode="json") for paper in papers], ensure_ascii=False)
