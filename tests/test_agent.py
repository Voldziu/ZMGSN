from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tool_eval.agent import decode_json_string_args
from tool_eval.tools.arguments import coerce_json_string_args
from tool_eval.tools.level1.formatting import format_apa_citation

SCHEMA: dict[str, Any] = {
    "authors": {"type": "array", "items": {"type": "string"}},
    "meta": {"anyOf": [{"type": "object"}, {"type": "null"}]},
    "year": {"type": "integer"},
    "title": {"type": "string"},
}


class _ToolCallingFake(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ToolCallingFake":
        return self


def test_stringified_list_is_decoded() -> None:
    args = coerce_json_string_args({"authors": '["Ada Lovelace", "Alan Turing"]'}, SCHEMA)
    assert args == {"authors": ["Ada Lovelace", "Alan Turing"]}


def test_stringified_optional_object_is_decoded() -> None:
    assert coerce_json_string_args({"meta": '{"k": 1}'}, SCHEMA) == {"meta": {"k": 1}}


@pytest.mark.parametrize(
    "args",
    [
        {"authors": "Ada Lovelace"},  # not JSON: left for validation to reject
        {"authors": '{"name": "Ada"}'},  # JSON of the wrong type
        {"title": '["looks", "like", "json"]'},  # declared as a string
        {"year": "2024"},  # scalars are left to pydantic's own coercion
        {"unknown": "[1]"},  # not in the schema
    ],
)
def test_other_values_pass_through(args: dict[str, Any]) -> None:
    assert coerce_json_string_args(args, SCHEMA) == args


def test_agent_accepts_stringified_list_argument() -> None:
    call = {
        "name": "format_apa_citation",
        "args": {
            "authors": '["Ada Lovelace"]',
            "year": "2024",
            "title": "Notes",
            "arxiv_id": "2401.00001",
        },
        "id": "call-1",
    }
    model = _ToolCallingFake(messages=iter([AIMessage("", tool_calls=[call]), AIMessage("done")]))
    agent = create_agent(
        model=model, tools=[format_apa_citation], middleware=[decode_json_string_args]
    )

    result = agent.invoke({"messages": [HumanMessage("cite")]})

    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.status == "success"
    assert "Lovelace, A." in str(tool_message.content)
