"""PDF download from arXiv and text extraction."""

import logging
from pathlib import Path

import httpx
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from tool_eval.exceptions import ExternalServiceError, InvalidInputError
from tool_eval.services.arxiv import validate_arxiv_id
from tool_eval.services.http import build_client

logger = logging.getLogger(__name__)

_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"


def download_arxiv_pdf(arxiv_id: str, *, dest: Path, timeout: float, max_bytes: int) -> None:
    """Download the PDF of an arXiv paper. Only arxiv.org URLs are ever requested.

    Args:
        arxiv_id: arXiv identifier; the URL is built from it, never taken from the caller.
        dest: Destination file path.
        timeout: HTTP timeout in seconds.
        max_bytes: Maximum accepted download size in bytes.

    Raises:
        InvalidInputError: If the identifier is malformed.
        ExternalServiceError: If the download fails, is too large or is not a PDF.
    """
    url = _PDF_URL.format(arxiv_id=validate_arxiv_id(arxiv_id))
    body = bytearray()
    try:
        with build_client(timeout=timeout) as client, client.stream("GET", url) as response:
            response.raise_for_status()
            for block in response.iter_bytes():
                body.extend(block)
                if len(body) > max_bytes:
                    msg = "PDF exceeds the size limit."
                    raise ExternalServiceError(msg)
    except httpx.HTTPError as exc:
        logger.warning("PDF download failed for %s: %s", arxiv_id, exc)
        msg = f"Could not download the PDF for {arxiv_id}."
        raise ExternalServiceError(msg) from exc
    if not body.startswith(b"%PDF"):
        msg = f"arXiv did not return a PDF for {arxiv_id}."
        raise ExternalServiceError(msg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(bytes(body))


def extract_pdf_text(path: Path) -> str:
    """Extract the text of every page of a PDF.

    Args:
        path: PDF file.

    Returns:
        Page texts joined by blank lines; empty if the PDF has no text layer.

    Raises:
        InvalidInputError: If the file is missing or cannot be parsed as a PDF.
    """
    if not path.is_file():
        msg = f"File not found: {path.name}"
        raise InvalidInputError(msg)
    try:
        reader = PdfReader(path)
        return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
    except (PyPdfError, ValueError, OSError) as exc:
        msg = f"Could not read PDF: {path.name}"
        raise InvalidInputError(msg) from exc
