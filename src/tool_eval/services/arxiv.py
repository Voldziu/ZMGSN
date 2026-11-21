"""arXiv search and metadata lookup via the public Atom API."""

import hashlib
import os
import random
import re
import tempfile
import threading
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from xml.etree.ElementTree import Element

import httpx
from defusedxml import ElementTree
from pydantic import BaseModel, ConfigDict

from tool_eval.config import get_settings
from tool_eval.exceptions import ExternalServiceError, InvalidInputError
from tool_eval.services.http import build_client
from tool_eval.services.network import count_network_request
from tool_eval.utils.logger import get_logger

logger = get_logger(__name__)

_API_URL = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_MIN_INTERVAL_SECONDS = 3.0  # arXiv API usage policy
_MAX_ATTEMPTS = 5
_BACKOFF_SECONDS = (5.0, 10.0, 20.0, 40.0)  # waits before attempts 2..5 without Retry-After
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_sleep = time.sleep  # patched in tests
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
    # Latest version number from the entry id (`…v3`); unknown for entries without one.
    version: int | None = None


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
    cache_dir: Path | None = None,
) -> list[ArxivPaper]:
    """Search arXiv, most relevant first, optionally within a submission-date window.

    Args:
        query: Free-text keywords.
        max_results: Maximum number of papers to return.
        date_from: Earliest submission date (inclusive), or `None`.
        date_to: Latest submission date (inclusive), or `None`.
        timeout: HTTP timeout in seconds.
        cache_dir: On-disk response cache; defaults to the active settings' `arxiv_cache_dir`.

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
        # Most relevant first; the date window above already limits submissions.
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    return _request(params, timeout=timeout, cache_dir=cache_dir)


def fetch_arxiv_papers(
    arxiv_ids: Sequence[str], *, timeout: float, cache_dir: Path | None = None
) -> list[ArxivPaper]:
    """Look up metadata for specific arXiv identifiers.

    Args:
        arxiv_ids: Identifiers to look up.
        timeout: HTTP timeout in seconds.
        cache_dir: On-disk response cache; defaults to the active settings' `arxiv_cache_dir`.

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
    papers = _request(
        {"id_list": ",".join(ids), "max_results": str(len(ids))},
        timeout=timeout,
        cache_dir=cache_dir,
    )
    found = {paper.arxiv_id for paper in papers}
    missing = [item for item in ids if _VERSION_SUFFIX.sub("", item) not in found]
    if missing:
        msg = f"arXiv identifiers not found: {', '.join(missing)}"
        raise InvalidInputError(msg)
    return papers


def _request(params: dict[str, str], *, timeout: float, cache_dir: Path | None) -> list[ArxivPaper]:
    """Answer from the on-disk cache, or ask arXiv (spaced, with retries) and cache the result."""
    directory = cache_dir if cache_dir is not None else get_settings().arxiv_cache_dir
    cache_file = directory / f"{_cache_key(params)}.xml" if directory is not None else None
    if cache_file is not None and cache_file.is_file():
        return _parse_feed(cache_file.read_text(encoding="utf-8"))
    count_network_request("arxiv_requests")
    xml_text = _fetch_with_retries(params, timeout=timeout)
    papers = _parse_feed(xml_text)  # raises on error entries, so failures are never cached
    if cache_file is not None:
        _write_atomically(cache_file, xml_text)
    return papers


def _fetch_with_retries(params: dict[str, str], *, timeout: float) -> str:
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        _throttle()
        retry_after: float | None = None
        try:
            with build_client(timeout=timeout) as client:
                response = client.get(_API_URL, params=params)
            if response.status_code not in _RETRY_STATUSES:
                response.raise_for_status()
                return response.text
            reason = f"HTTP {response.status_code}"
            retry_after = _retry_after_seconds(response)
        except httpx.HTTPStatusError as exc:
            logger.warning("arXiv request failed: %s", exc)
            msg = f"arXiv request failed (HTTP {exc.response.status_code})."
            raise ExternalServiceError(msg) from exc
        except httpx.TransportError as exc:  # timeouts and connection errors
            reason = type(exc).__name__
        if attempt == _MAX_ATTEMPTS:
            logger.warning("arXiv request failed after %d attempts: %s", attempt, reason)
            break
        wait = retry_after if retry_after is not None else _BACKOFF_SECONDS[attempt - 1]
        wait += random.uniform(0, 1)  # noqa: S311 - jitter, not security
        logger.warning("arXiv request failed (%s); retrying in %.0f s", reason, wait)
        _sleep(wait)
    msg = f"arXiv request failed after {_MAX_ATTEMPTS} attempts."
    raise ExternalServiceError(msg)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After", "")
    return float(value) if value.strip().isdigit() else None


def _cache_key(params: dict[str, str]) -> str:
    canonical = "&".join(f"{key}={params[key]}" for key in sorted(params))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    Path(tmp).replace(path)


def _throttle() -> None:
    global _last_request_at
    with _throttle_lock:
        wait = _MIN_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            _sleep(wait)
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
    versioned = raw_id.split("/abs/", 1)[1]
    arxiv_id = _VERSION_SUFFIX.sub("", versioned)
    suffix = _VERSION_SUFFIX.search(versioned)
    return ArxivPaper(
        version=int(suffix.group()[1:]) if suffix else None,
        arxiv_id=arxiv_id,
        title=_text(entry, "title"),
        authors=[_text(author, "name") for author in entry.findall(f"{_ATOM}author")],
        published=date.fromisoformat(_text(entry, "published")[:10]),
        abstract=_text(entry, "summary"),
    )


def _text(element: Element, tag: str) -> str:
    child = element.find(f"{_ATOM}{tag}")
    return " ".join((child.text or "").split()) if child is not None else ""
