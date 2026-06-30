# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm.renderers import ChatParams
from vllm.renderers.rwkv import RWKVRenderer
from vllm.sampling_params import SamplingParams
from vllm.tokenizers.rwkv import RWKVTokenizer


def _write_vocab(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "1 'S' 1",
                "2 'y' 1",
                "3 's' 1",
                "4 't' 1",
                "5 'e' 1",
                "6 'm' 1",
                "7 ':' 1",
                "8 ' ' 1",
                "9 'h' 1",
                "10 'i' 1",
                "11 '\\n\\n' 2",
                "12 'U' 1",
                "13 'r' 1",
                "14 '<|endoftext|>' 13",
                "15 '<|im_start|>' 12",
                "16 '<|im_end|>' 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@dataclass
class MockHFConfig:
    model_type: str = "rwkv7"


@dataclass
class MockModelConfig:
    runner_type = "generate"
    model: str = "native-rwkv7"
    tokenizer: str = "native-rwkv7"
    trust_remote_code: bool = False
    max_model_len: int = 128
    tokenizer_revision = None
    tokenizer_mode = "rwkv"
    hf_config = MockHFConfig()
    encoder_config: dict[str, Any] | None = None
    multimodal_config = None
    allowed_local_media_path = None
    allowed_media_domains = None
    enable_prompt_embeds: bool = True
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


def test_rwkv_renderer_renders_chat_messages(tmp_path):
    vocab_path = tmp_path / "rwkv_vocab_v20250609.txt"
    _write_vocab(vocab_path)
    tokenizer = RWKVTokenizer.from_pretrained(vocab_path)
    renderer = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(tokenizer=str(vocab_path)),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=tokenizer,
    )

    conversation, prompt = renderer.render_messages(
        [
            {"role": "system", "content": "hi"},
            {"role": "user", "content": "hi"},
        ],
        ChatParams(chat_template_kwargs={"add_generation_prompt": True}),
    )

    assert [message["role"] for message in conversation] == ["system", "user"]
    assert "prompt" in prompt
    assert prompt["prompt"].startswith("System: hi\n\nUser: hi\n\nAssistant:")


def test_rwkv_renderer_normalizes_tool_history_before_template():
    class CaptureTokenizer:
        bos_token_id = None
        eos_token_id = None
        chat_template = "template"

        def __init__(self) -> None:
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            self.messages = messages
            return "prompt"

        def encode(self, text: str, **kwargs):
            del kwargs
            return [ord(char) for char in text]

    tokenizer = CaptureTokenizer()
    renderer = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=tokenizer,
    )

    conversation, prompt = renderer.render_messages(
        [
            {"role": "user", "content": "北京天气怎么样"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query": "北京天气"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather",
                "content": "今天白天有雷阵雨，夜晚有小雨。",
            },
        ],
        ChatParams(chat_template_kwargs={"add_generation_prompt": True}),
    )

    assert prompt["prompt"] == "prompt"
    assert tokenizer.messages == conversation
    assert conversation[1]["tool_calls"][0]["function"]["arguments"] == {
        "query": "北京天气"
    }
    assert conversation[2]["name"] == "search_web"


def test_rwkv_renderer_overrides_generation_eos_tokens(tmp_path):
    vocab_path = tmp_path / "rwkv_vocab_v20250609.txt"
    _write_vocab(vocab_path)
    tokenizer = RWKVTokenizer.from_pretrained(vocab_path)
    renderer = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(tokenizer=str(vocab_path)),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=tokenizer,
    )

    generation_config = renderer.get_generation_config_fields({"eos_token_id": 2})

    assert generation_config["eos_token_id"] == [16, 14]

    sampling_params = SamplingParams()
    sampling_params.update_from_generation_config(
        generation_config,
        renderer.get_eos_token_id(),
    )

    assert sampling_params.stop_token_ids == [16]
    assert sampling_params.all_stop_token_ids == {14, 16}
