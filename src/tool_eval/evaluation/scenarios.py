"""Loading of scenario definitions from a JSON file kept outside the code."""

from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from tool_eval.evaluation.models import Scenario
from tool_eval.exceptions import InvalidInputError

_ADAPTER = TypeAdapter(list[Scenario])


def load_scenarios(path: Path) -> list[Scenario]:
    """Load and validate scenarios.

    Args:
        path: JSON file containing a list of scenarios.

    Returns:
        Scenarios in file order.

    Raises:
        InvalidInputError: If the file is missing, malformed or contains duplicate ids.
    """
    try:
        scenarios = _ADAPTER.validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        msg = f"Cannot load scenarios from {path.name}: {exc}"
        raise InvalidInputError(msg) from exc
    ids = [scenario.id for scenario in scenarios]
    if len(set(ids)) != len(ids):
        msg = "Scenario ids must be unique."
        raise InvalidInputError(msg)
    return scenarios
