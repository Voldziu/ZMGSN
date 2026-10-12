"""Wikipedia search via the MediaWiki Action API."""

import logging

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from tool_eval.exceptions import ExternalServiceError, InvalidInputError
from tool_eval.services.http import build_client

logger = logging.getLogger(__name__)

_API_URL = "https://en.wikipedia.org/w/api.php"


class WikipediaResult(BaseModel):
    """One Wikipedia search hit with an introductory extract."""

    model_config = ConfigDict(frozen=True)

    title: str
    extract: str
    url: str


class _Page(BaseModel):
    pageid: int
    title: str
    extract: str = ""
    index: int = 0


class _Query(BaseModel):
    pages: dict[str, _Page] = {}


class _Response(BaseModel):
    query: _Query | None = None


def search_wikipedia(query: str, *, max_results: int, timeout: float) -> list[WikipediaResult]:
    """Search English Wikipedia and return short introductory extracts.

    Args:
        query: Search terms.
        max_results: Maximum number of results.
        timeout: HTTP timeout in seconds.

    Returns:
        Results ordered by relevance, possibly empty.

    Raises:
        InvalidInputError: If the query is empty.
        ExternalServiceError: If Wikipedia is unreachable or returns unusable data.
    """
    if not query.strip():
        msg = "Query must not be empty."
        raise InvalidInputError(msg)
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": str(max_results),
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "exsentences": "3",
        "exlimit": "max",
    }
    try:
        with build_client(timeout=timeout) as client:
            response = client.get(_API_URL, params=params)
            response.raise_for_status()
        parsed = _Response.model_validate(response.json())
    except (httpx.HTTPError, ValueError, ValidationError) as exc:
        logger.warning("Wikipedia request failed: %s", exc)
        msg = "Wikipedia request failed."
        raise ExternalServiceError(msg) from exc
    pages = sorted(parsed.query.pages.values(), key=lambda page: page.index) if parsed.query else []
    return [
        WikipediaResult(
            title=page.title,
            extract=page.extract.strip(),
            url=f"https://en.wikipedia.org/?curid={page.pageid}",
        )
        for page in pages
    ]
