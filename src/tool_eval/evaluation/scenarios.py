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
        InvalidInputError: If the file is missing, malformed or contains duplicate ids, or a
            variant does not match its base scenario.
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
    _check_variants(scenarios)
    return scenarios


def _check_variants(scenarios: list[Scenario]) -> None:
    """A variant points to a base scenario with the same topic and window, and is named after it."""
    bases = {s.id: s for s in scenarios if s.task_type == "base"}
    for variant in (s for s in scenarios if s.task_type == "variant"):
        base = bases.get(variant.base_id)
        if base is None:
            msg = f"Variant {variant.id}: base scenario {variant.base_id!r} does not exist"
            raise InvalidInputError(msg)
        if (variant.topic, variant.date_from, variant.date_to) != (
            base.topic,
            base.date_from,
            base.date_to,
        ):
            msg = f"Variant {variant.id}: topic and window must equal those of {base.id}"
            raise InvalidInputError(msg)
        if variant.id != f"{base.id}-{variant.requirement}":
            msg = f"Variant {variant.id}: id must be {base.id}-{variant.requirement}"
            raise InvalidInputError(msg)
