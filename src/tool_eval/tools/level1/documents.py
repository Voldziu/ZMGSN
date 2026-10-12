"""Document tools: download a PDF, extract its text, split the text into chunks."""

import json

from langchain_core.tools import tool

from tool_eval.config import get_settings
from tool_eval.services import papers


@tool(parse_docstring=True)
def download_pdf(arxiv_id: str) -> str:
    """Download the PDF of an arXiv paper into the workspace.

    Args:
        arxiv_id: arXiv identifier, for example 2401.00001.

    Returns:
        Workspace-relative path of the downloaded PDF.
    """
    return papers.download_paper_pdf(arxiv_id, settings=get_settings())


@tool(parse_docstring=True)
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract the text of a PDF in the workspace and save it as a text file.

    Args:
        pdf_path: Workspace-relative path of the PDF.

    Returns:
        Workspace-relative path of the text file.
    """
    return papers.extract_paper_text(pdf_path, settings=get_settings())


@tool(parse_docstring=True)
def split_text_into_chunks(text_path: str) -> str:
    """Split a text file into smaller chunk files that fit into a prompt.

    Only the leading chunks of a long document are kept.

    Args:
        text_path: Workspace-relative path of the text file.

    Returns:
        JSON list of workspace-relative chunk file paths, in document order.
    """
    return json.dumps(papers.split_paper_text(text_path, settings=get_settings()))
