"""Formatting tools: APA citations and Markdown tables."""

from langchain_core.tools import tool

from tool_eval.services import formatting


@tool(parse_docstring=True)
def format_apa_citation(authors: list[str], year: int, title: str, arxiv_id: str) -> str:
    """Format an arXiv paper as an APA 7 reference.

    Args:
        authors: Full author names in "First Last" order, as returned by search_arxiv.
        year: Publication year.
        title: Paper title.
        arxiv_id: arXiv identifier.

    Returns:
        The formatted reference.
    """
    return formatting.format_apa_citation(authors, year=year, title=title, arxiv_id=arxiv_id)


@tool(parse_docstring=True)
def create_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a Markdown table.

    Args:
        headers: Column headers.
        rows: Table rows; each row needs one cell per header.

    Returns:
        The Markdown table text.
    """
    return formatting.format_markdown_table(headers, rows)
