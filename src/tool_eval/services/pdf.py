"""PDF download from arXiv and text extraction."""

from pathlib import Path

import httpx
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from tool_eval.exceptions import ExternalServiceError, InvalidInputError
from tool_eval.services.arxiv import validate_arxiv_id
from tool_eval.services.http import build_client
from tool_eval.utils.logger import get_logger

logger = get_logger(__name__)

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
    too_large = f"The PDF of {arxiv_id} exceeds the {max_bytes / 1_000_000:g} MB size limit."
    body = bytearray()
    try:
        with build_client(timeout=timeout) as client, client.stream("GET", url) as response:
            response.raise_for_status()
            declared = response.headers.get("Content-Length", "")
            if declared.isdigit() and int(declared) > max_bytes:
                raise ExternalServiceError(too_large)
            for block in response.iter_bytes():
                body.extend(block)
                if len(body) > max_bytes:
                    raise ExternalServiceError(too_large)
    except httpx.HTTPStatusError as exc:
        logger.warning("PDF download failed for %s: %s", arxiv_id, exc)
        msg = f"arXiv returned HTTP {exc.response.status_code} for the PDF of {arxiv_id}."
        raise ExternalServiceError(msg) from exc
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
        Page texts joined by blank lines, always valid UTF-8 (unencodable characters such as lone
        surrogates become `?`); empty if the PDF has no text layer.

    Raises:
        InvalidInputError: If the file is missing or cannot be parsed as a PDF.
    """
    if not path.is_file():
        msg = f"File not found: {path.name}"
        raise InvalidInputError(msg)
    try:
        reader = PdfReader(path)
        text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
    except (PyPdfError, ValueError, OSError) as exc:
        msg = f"Could not read PDF: {path.name}"
        raise InvalidInputError(msg) from exc
    # pypdf can emit lone UTF-16 surrogates for unmapped glyphs; they cannot be written as UTF-8.
    return text.encode("utf-8", "replace").decode("utf-8")
