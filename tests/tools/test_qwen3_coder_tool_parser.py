import json

import pytest
from psrl.tools.tool_parser import Qwen3XMLToolParser
from psrl.tools.tool_parser.base import ToolParser

pytestmark = pytest.mark.cpu_test


class _Tokenizer:
    def __init__(self, text: str):
        self.text = text

    def decode(self, token_ids: list[int]) -> str:
        return self.text


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "exact": {"type": "boolean"},
                    "filters": {"type": "object"},
                },
            },
        },
    }
]


def test_qwen3_coder_parser_is_registered():
    parser = ToolParser.get_tool_parser("qwen3_coder", _Tokenizer("plain text"))
    assert isinstance(parser, Qwen3XMLToolParser)


def test_qwen35_xml_tool_call_uses_schema_types():
    text = (
        "I will search."
        "<tool_call><function=search>"
        "<parameter=query>hybrid attention</parameter>"
        "<parameter=limit>3</parameter>"
        "<parameter=exact>true</parameter>"
        '<parameter=filters>{"year": 2026}</parameter>'
        "</function></tool_call>"
    )
    parser = Qwen3XMLToolParser(_Tokenizer(text))

    content, calls = parser.extract_tool_calls([1, 2, 3], tools=TOOLS)

    assert content == "I will search."
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert json.loads(calls[0].arguments) == {
        "query": "hybrid attention",
        "limit": 3,
        "exact": True,
        "filters": {"year": 2026},
    }


def test_qwen35_xml_tool_call_accepts_truncated_closing_tags():
    text = "<tool_call><function=search><parameter=query>qwen3.5</parameter>"
    parser = Qwen3XMLToolParser(_Tokenizer(text))

    content, calls = parser.extract_tool_calls([1], tools=TOOLS)

    assert content == ""
    assert len(calls) == 1
    assert json.loads(calls[0].arguments) == {"query": "qwen3.5"}


def test_qwen35_xml_tool_call_returns_text_when_no_call_exists():
    parser = Qwen3XMLToolParser(_Tokenizer("ordinary response"))

    assert parser.extract_tool_calls([1], tools=TOOLS) == ("ordinary response", [])
