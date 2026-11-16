"""Normalisation of tool-call arguments before the tool validates them."""

import ast
import json
from collections.abc import Mapping
from typing import Any

_DECODABLE = {"array": list, "object": dict}


def coerce_json_string_args(args: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    r"""Decode list/object arguments that the model sent as JSON-encoded strings.

    Some models (Llama 3.1 through vLLM's `llama3_json` parser) serialise nested values as
    strings, e.g. `"authors": "[\"A\", \"B\"]"` or the Python literal `"['A', 'B']"`. Pydantic
    coerces scalar strings itself but rejects strings for lists and objects. A string is replaced
    only when the argument is declared as an array or object and the string decodes (as JSON,
    else as a Python literal) to exactly that type; everything else is returned unchanged, so
    genuinely wrong arguments still fail validation.

    Args:
        args: Arguments of one tool call.
        schema: The tool's per-argument JSON schemas (`BaseTool.args`).

    Returns:
        A new dict with the decodable strings replaced.
    """
    result = dict(args)
    for name, value in args.items():
        if not isinstance(value, str):
            continue
        wanted = _declared_types(schema.get(name, {})) & _DECODABLE.keys()
        if not wanted:
            continue
        decoded = _decode(value)
        if any(isinstance(decoded, _DECODABLE[kind]) for kind in wanted):
            result[name] = decoded
    return result


def _decode(text: str) -> Any:
    """Decode JSON, else a Python literal (`ast.literal_eval` evaluates literals only, no code)."""
    try:
        return json.loads(text)
    except ValueError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _declared_types(schema: Mapping[str, Any]) -> set[str]:
    declared = schema.get("type")
    types = {declared} if isinstance(declared, str) else set(declared or ())
    for key in ("anyOf", "oneOf"):
        for option in schema.get(key, ()):
            types |= _declared_types(option)
    return types
