"""arXiv search and metadata lookup via the public Atom API."""

import logging
import re
import threading
import time
from collections.abc import Sequence
from datetime import date
from xml.etree.ElementTree import Element

import httpx
from defusedxml import ElementTree
from pydantic import BaseModel, ConfigDict

from tool_eval.exceptions import ExternalServiceError, InvalidInputError
from tool_eval.services.http import build_client

logger = logging.getLogger(__name__)

_API_URL = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_MIN_INTERVAL_SECONDS = 3.0  # arXiv API usage policy
_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7})(v\d+)?")
_VERSION_SUFFIX = re.compile(r"v\d+$")

_throttle_lock = threading.Lock()
_last_request_at = 0.0


class ArxivPaper(BaseModel):
    """Metadata of one arXiv paper."""

    model_config = ConfigDict(frozen=True)

    arxiv_id: str
    title: str
    authors: list[str]
    published: date
    abstract: str


def validate_arxiv_id(arxiv_id: str) -> str:
    """Validate and normalize an arXiv identifier.

    Args:
        arxiv_id: Identifier such as `2401.00001`, `2401.00001v2` or `hep-th/9901001`.

    Returns:
        The identifier with surrounding whitespace removed.

    Raises:
        InvalidInputError: If the identifier has an unsupported format.
    """
    cleaned = arxiv_id.strip()
    if not _ID_PATTERN.fullmatch(cleaned):
        msg = f"Invalid arXiv identifier: {cleaned!r}"
        raise InvalidInputError(msg)
    return cleaned


def search_arxiv(
    query: str,
    *,
    max_results: int,
    date_from: date | None,
    date_to: date | None,
    timeout: float,
) -> list[ArxivPaper]:
    """Search arXiv, newest submissions first, optionally within a submission-date window.

    Args:
        query: Free-text keywords.
        max_results: Maximum number of papers to return.
        date_from: Earliest submission date (inclusive), or `None`.
        date_to: Latest submission date (inclusive), or `None`.
        timeout: HTTP timeout in seconds.

    Returns:
        Matching papers, possibly empty.

    Raises:
        InvalidInputError: If the query contains no searchable terms.
        ExternalServiceError: If arXiv is unreachable or returns unusable data.
    """
    terms = re.findall(r"[\w-]+", query)
    if not terms:
        msg = "Query contains no searchable terms."
        raise InvalidInputError(msg)
    search_query = " AND ".join(f"all:{term}" for term in terms)
    if date_from is not None or date_to is not None:
        start = (date_from or date(1991, 1, 1)).strftime("%Y%m%d")
        end = (date_to or date.today()).strftime("%Y%m%d")
        search_query += f" AND submittedDate:[{start}0000 TO {end}2359]"
    params = {
        "search_query": search_query,
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return _request(params, timeout=timeout)


def fetch_arxiv_papers(arxiv_ids: Sequence[str], *, timeout: float) -> list[ArxivPaper]:
    """Look up metadata for specific arXiv identifiers.

    Args:
        arxiv_ids: Identifiers to look up.
        timeout: HTTP timeout in seconds.

    Returns:
        Papers found, in arXiv's order.

    Raises:
        InvalidInputError: If an identifier is malformed or some identifiers were not found.
        ExternalServiceError: If arXiv is unreachable or returns unusable data.
    """
    ids = [validate_arxiv_id(item) for item in arxiv_ids]
    if not ids:
        msg = "At least one arXiv identifier is required."
        raise InvalidInputError(msg)
    papers = _request({"id_list": ",".join(ids), "max_results": str(len(ids))}, timeout=timeout)
    found = {paper.arxiv_id for paper in papers}
    missing = [item for item in ids if _VERSION_SUFFIX.sub("", item) not in found]
    if missing:
        msg = f"arXiv identifiers not found: {', '.join(missing)}"
        raise InvalidInputError(msg)
    return papers


def _request(params: dict[str, str], *, timeout: float) -> list[ArxivPaper]:
    _throttle()
    try:
        with build_client(timeout=timeout) as client:
            response = client.get(_API_URL, params=params)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("arXiv request failed: %s", exc)
        msg = "arXiv request failed."
        raise ExternalServiceError(msg) from exc
    return _parse_feed(response.text)


def _throttle() -> None:
    global _last_request_at
    with _throttle_lock:
        wait = _MIN_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _parse_feed(xml_text: str) -> list[ArxivPaper]:
    try:
        root = ElementTree.fromstring(xml_text)
        return [_parse_entry(entry) for entry in root.findall(f"{_ATOM}entry")]
    except (ElementTree.ParseError, ValueError) as exc:
        msg = "arXiv returned a malformed response."
        raise ExternalServiceError(msg) from exc


def _parse_entry(entry: Element) -> ArxivPaper:
    raw_id = _text(entry, "id")
    if "/abs/" not in raw_id:
        msg = f"arXiv returned an error entry: {_text(entry, 'summary')[:120]}"
        raise ExternalServiceError(msg)
    arxiv_id = _VERSION_SUFFIX.sub("", raw_id.split("/abs/", 1)[1])
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=_text(entry, "title"),
        authors=[_text(author, "name") for author in entry.findall(f"{_ATOM}author")],
        published=date.fromisoformat(_text(entry, "published")[:10]),
        abstract=_text(entry, "summary"),
    )


def _text(element: Element, tag: str) -> str:
    child = element.find(f"{_ATOM}{tag}")
    return " ".join((child.text or "").split()) if child is not None else ""
