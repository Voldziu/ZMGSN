"""Pure formatting helpers: APA citations and Markdown tables."""

from collections.abc import Sequence

from tool_eval.exceptions import InvalidInputError

_MAX_LISTED_AUTHORS = 20


def format_apa_citation(authors: Sequence[str], *, year: int, title: str, arxiv_id: str) -> str:
    """Format an arXiv preprint as an APA 7 reference.

    Args:
        authors: Full author names in "First Last" order.
        year: Publication year.
        title: Paper title.
        arxiv_id: arXiv identifier used to build the URL.

    Returns:
        The formatted reference.

    Raises:
        InvalidInputError: If no authors are given.
    """
    if not authors:
        msg = "At least one author is required."
        raise InvalidInputError(msg)
    names = [_apa_author(author) for author in authors]
    if len(names) > _MAX_LISTED_AUTHORS:
        author_text = ", ".join(names[: _MAX_LISTED_AUTHORS - 1]) + f", ... {names[-1]}"
    elif len(names) == 1:
        author_text = names[0]
    else:
        author_text = ", ".join(names[:-1]) + f", & {names[-1]}"
    return f"{author_text} ({year}). {title.strip()}. arXiv. https://arxiv.org/abs/{arxiv_id}"


def format_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a GitHub-flavoured Markdown table.

    Args:
        headers: Column headers.
        rows: Table rows; each must have one cell per header.

    Returns:
        The Markdown table.

    Raises:
        InvalidInputError: If there are no headers or a row has the wrong number of cells.
    """
    if not headers:
        msg = "At least one header is required."
        raise InvalidInputError(msg)
    lines = [_row(headers), _row(["---"] * len(headers))]
    for index, row in enumerate(rows, start=1):
        if len(row) != len(headers):
            msg = f"Row {index} has {len(row)} cells; expected {len(headers)}."
            raise InvalidInputError(msg)
        lines.append(_row(row))
    return "\n".join(lines)


def _row(cells: Sequence[str]) -> str:
    escaped = [cell.replace("|", "\\|").replace("\n", " ").strip() for cell in cells]
    return "| " + " | ".join(escaped) + " |"


def _apa_author(full_name: str) -> str:
    parts = full_name.split()
    if len(parts) < 2:
        return full_name.strip()
    initials = " ".join(f"{part[0]}." for part in parts[:-1])
    return f"{parts[-1]}, {initials}"
