"""Registry of tool sets. Add a new granularity level by adding one entry to `TOOLSETS`."""

from collections.abc import Mapping, Sequence

from langchain_core.tools import BaseTool

from tool_eval.exceptions import InvalidInputError
from tool_eval.tools import level1, level2, level3

TOOLSETS: Mapping[str, Sequence[BaseTool]] = {
    "level1": level1.TOOLS,
    "level2": level2.TOOLS,
    "level3": level3.TOOLS,
}


def get_tools(level: str) -> list[BaseTool]:
    """Return the tools for one granularity level.

    Args:
        level: Key of `TOOLSETS`, for example `level1`.

    Returns:
        A new list with the level's tools.

    Raises:
        InvalidInputError: If the level is unknown.
    """
    try:
        return list(TOOLSETS[level])
    except KeyError as exc:
        msg = f"Unknown tool level {level!r}; available: {', '.join(TOOLSETS)}"
        raise InvalidInputError(msg) from exc
