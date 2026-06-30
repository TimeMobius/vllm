# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.tokenizers import cached_get_tokenizer
from vllm.tokenizers.rwkv import RWKVTokenizer

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams


class RWKVRenderer(BaseRenderer[RWKVTokenizer]):
    _DEFAULT_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")

    @staticmethod
    def _fill_tool_message_names(conversation: list[ConversationMessage]) -> None:
        tool_call_names: dict[str, str] = {}
        for message in conversation:
            if message["role"] == "assistant":
                tool_calls = message.get("tool_calls")
                if not isinstance(tool_calls, list):
                    continue

                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    tool_call_id = tool_call.get("id")
                    function = tool_call.get("function")
                    if (
                        isinstance(tool_call_id, str)
                        and isinstance(function, dict)
                        and isinstance(function.get("name"), str)
                    ):
                        tool_call_names[tool_call_id] = function["name"]
            elif message["role"] == "tool" and not message.get("name"):
                tool_call_id = message.get("tool_call_id")
                if isinstance(tool_call_id, str) and tool_call_id in tool_call_names:
                    message["name"] = tool_call_names[tool_call_id]

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        config: VllmConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> "RWKVRenderer":
        model_config = config.model_config
        if model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = cached_get_tokenizer(
                tokenizer_cls=RWKVTokenizer,
                **tokenizer_kwargs,
            )

        return cls(config, tokenizer)

    def get_generation_config_fields(
        self, generation_config_fields: dict[str, Any]
    ) -> dict[str, Any]:
        tokenizer = self.tokenizer
        if tokenizer is None:
            return generation_config_fields

        stop_token_ids: list[int] = []
        for token in self._DEFAULT_STOP_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id not in stop_token_ids:
                stop_token_ids.append(token_id)

        if not stop_token_ids:
            return generation_config_fields

        updated_generation_config = dict(generation_config_fields)
        updated_generation_config["eos_token_id"] = stop_token_ids
        return updated_generation_config

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        tokenizer = self.get_tokenizer()
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        self._fill_tool_message_names(conversation)

        prompt_raw = tokenizer.apply_chat_template(
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        tokenizer = self.get_tokenizer()
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        self._fill_tool_message_names(conversation)

        prompt_raw = tokenizer.apply_chat_template(
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
