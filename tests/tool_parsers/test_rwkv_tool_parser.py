# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.parser.parser_manager import ParserManager
from vllm.tool_parsers import ToolParserManager


class SimpleTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]


@pytest.fixture()
def parser():
    parser_cls = ToolParserManager.get_tool_parser("rwkv")
    return parser_cls(SimpleTokenizer())


@pytest.fixture()
def delegating_parser():
    parser_cls = ParserManager.get_parser(
        tool_parser_name="rwkv",
        reasoning_parser_name="rwkv",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    return parser_cls(SimpleTokenizer())


@pytest.fixture()
def weather_tool():
    return ChatCompletionToolsParam(
        type="function",
        function=FunctionDefinition(
            name="get_weather",
            description="Get weather.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "rain": {"type": "boolean"},
                },
            },
        ),
    )


@pytest.fixture()
def chat_request(weather_tool):
    return ChatCompletionRequest(
        model="rwkv-test",
        messages=[],
        tools=[weather_tool],
        tool_choice="auto",
    )


def _tool_xml(*, content: str = "") -> str:
    return (
        f"{content}<tool_call>\n"
        '<invoke name="get_weather">\n'
        '<parameter name="city">杭州</parameter>\n'
        '<parameter name="days">2</parameter>\n'
        '<parameter name="rain">true</parameter>\n'
        "</invoke>\n"
        "</tool_call>"
    )


def test_rwkv_tool_parser_is_registered(parser):
    assert parser.tool_call_start_token == "<tool_call>"
    assert parser.tool_call_end_token == "</tool_call>"
    assert parser.supports_required_and_named is False


def test_adjust_request_keeps_sp_tool_markers(parser, chat_request):
    assert chat_request.skip_special_tokens is True

    adjusted_request = parser.adjust_request(chat_request)

    assert adjusted_request.skip_special_tokens is False


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "get_weather"}},
    ],
)
def test_adjust_request_skips_json_guidance_for_native_choices(
    parser, weather_tool, tool_choice
):
    request = ChatCompletionRequest(
        model="rwkv-test",
        messages=[],
        tools=[weather_tool.model_dump()],
        tool_choice=tool_choice,
    )

    adjusted_request = parser.adjust_request(request)

    assert adjusted_request.structured_outputs is None
    assert adjusted_request.skip_special_tokens is False


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "get_weather"}},
    ],
)
def test_required_and_named_choices_use_rwkv_xml_parser(
    delegating_parser, weather_tool, tool_choice
):
    request = ChatCompletionRequest(
        model="rwkv-test",
        messages=[],
        tools=[weather_tool.model_dump()],
        tool_choice=tool_choice,
    )

    reasoning, content, tool_calls = delegating_parser.parse(
        "<think>planning</think>" + _tool_xml(content="I'll check.\n"),
        request,
        enable_auto_tools=True,
    )

    assert reasoning == "planning"
    assert content == "I'll check.\n"
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert json.loads(tool_calls[0].arguments) == {
        "city": "杭州",
        "days": 2,
        "rain": True,
    }


def test_required_choice_uses_rwkv_xml_parser_for_streaming(
    delegating_parser, weather_tool
):
    request = ChatCompletionRequest(
        model="rwkv-test",
        messages=[],
        tools=[weather_tool],
        tool_choice="required",
    )
    deltas = [
        "<think>planning</think>I'll check.\n<tool_call>\n",
        '<invoke name="get_weather">\n',
        '<parameter name="city">杭州</parameter>\n',
        '<parameter name="days">2</parameter>\n',
        '<parameter name="rain">true</parameter>\n',
        "</invoke>\n",
        "</tool_call>",
    ]

    messages = []
    for index, delta in enumerate(deltas):
        message = delegating_parser.parse_delta(
            delta,
            delegating_parser.model_tokenizer.encode(delta),
            request,
            finished=index == len(deltas) - 1,
        )
        if message is not None:
            messages.append(message)

    tool_messages = [message for message in messages if message.tool_calls]
    assert len(tool_messages) == 1
    assert tool_messages[0].content is None
    assert len(tool_messages[0].tool_calls) == 1
    call = tool_messages[0].tool_calls[0]
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {
        "city": "杭州",
        "days": 2,
        "rain": True,
    }


def test_extract_tool_calls_non_streaming(parser, chat_request):
    result = parser.extract_tool_calls(
        _tool_xml(content="I'll check.\n"),
        chat_request,
    )

    assert result.tools_called is True
    assert result.content == "I'll check.\n"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {
        "city": "杭州",
        "days": 2,
        "rain": True,
    }


def test_extract_multiple_tool_calls_non_streaming(parser, chat_request):
    second = (
        "<tool_call>\n"
        '<invoke name="get_weather">\n'
        '<parameter name="city">上海</parameter>\n'
        "</invoke>\n"
        "</tool_call>"
    )

    result = parser.extract_tool_calls(_tool_xml() + "\n" + second, chat_request)

    assert result.tools_called is True
    assert [call.function.name for call in result.tool_calls] == [
        "get_weather",
        "get_weather",
    ]
    assert json.loads(result.tool_calls[1].function.arguments) == {"city": "上海"}


def test_extract_tool_calls_returns_content_without_call(parser, chat_request):
    result = parser.extract_tool_calls("plain answer", chat_request)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == "plain answer"


def _stream(parser, chat_request, deltas: list[str]) -> list[DeltaMessage]:
    previous_text = ""
    previous_ids: list[int] = []
    messages: list[DeltaMessage] = []
    tokenizer = parser.model_tokenizer

    for delta in deltas:
        delta_ids = tokenizer.encode(delta)
        current_text = previous_text + delta
        current_ids = previous_ids + delta_ids
        message = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta,
            previous_ids,
            current_ids,
            delta_ids,
            chat_request,
        )
        if message is not None:
            messages.append(message)
        previous_text = current_text
        previous_ids = current_ids

    return messages


def test_streaming_passes_content_before_tool_call(parser, chat_request):
    messages = _stream(parser, chat_request, ["I'll check.\n", "<tool", "_call>"])

    assert len(messages) == 1
    assert messages[0].content == "I'll check.\n"
    assert messages[0].tool_calls == []


def test_streaming_emits_tool_call_when_invoke_completes(parser, chat_request):
    messages = _stream(
        parser,
        chat_request,
        [
            "I'll check.\n<tool_call>\n",
            '<invoke name="get_weather">\n',
            '<parameter name="city">杭州</parameter>\n',
            '<parameter name="days">2</parameter>\n',
            '<parameter name="rain">true</parameter>\n',
            "</invoke>\n",
            "</tool_call>",
        ],
    )

    tool_messages = [message for message in messages if message.tool_calls]
    assert len(tool_messages) == 1
    assert tool_messages[0].content is None
    delta_call = tool_messages[0].tool_calls[0]
    assert delta_call.index == 0
    assert delta_call.type == "function"
    assert delta_call.function.name == "get_weather"
    assert json.loads(delta_call.function.arguments) == {
        "city": "杭州",
        "days": 2,
        "rain": True,
    }


def test_streaming_emits_each_invoke_once(parser, chat_request):
    complete = _tool_xml()
    messages = _stream(parser, chat_request, [complete, "\ntrailing"])

    tool_calls = [tool_call for message in messages for tool_call in message.tool_calls]
    assert len(tool_calls) == 1
    assert tool_calls[0].index == 0
