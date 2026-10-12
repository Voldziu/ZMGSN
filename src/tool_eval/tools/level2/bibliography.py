"""Level 2 tool: build a bibliography file from arXiv identifiers."""

from langchain_core.tools import tool

from tool_eval.config import get_settings
from tool_eval.services import reports


@tool(parse_docstring=True)
def generate_bibliography(arxiv_ids: list[str], output_path: str = "bibliography.md") -> str:
    """Look up papers on arXiv and write an alphabetical APA bibliography file.

    Args:
        arxiv_ids: arXiv identifiers to include.
        output_path: Workspace-relative .md or .txt file to write.

    Returns:
        Workspace-relative path of the bibliography file.
    """
    return reports.write_bibliography(arxiv_ids, output_ref=output_path, settings=get_settings())


@tool(parse_docstring=True)
def append_bibliography(arxiv_ids: list[str]) -> str:
    """Append a "## Bibliography" section with alphabetical APA references to reviews/review.md.

    Args:
        arxiv_ids: arXiv identifiers of the reviewed papers.

    Returns:
        Workspace-relative path of the review file.
    """
    return reports.append_bibliography(arxiv_ids, settings=get_settings())
