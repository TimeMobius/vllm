# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.registry import resolve_tokenizer_args


def _write_vocab(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "1 'a' 1",
                "2 'b' 1",
                "3 'ab' 2",
                "4 '\\n\\n' 2",
                "5 '<|endoftext|>' 13",
                "6 '<|im_start|>' 12",
                "7 '<|im_end|>' 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_hf_rwkv_tokenizer_dir(path: Path) -> Path:
    vocab_path = path / "rwkv_vocab_v20230424.txt"
    vocab_path.write_text(
        "\n".join(
            [
                "1 'a' 1",
                "2 'b' 1",
                "3 '\\n\\n' 2",
                "4 'U' 1",
                "5 's' 1",
                "6 'e' 1",
                "7 'r' 1",
                "8 ':' 1",
                "9 ' ' 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "hf_rwkv_tokenizer.py").write_text("", encoding="utf-8")
    (path / "added_tokens.json").write_text(
        json.dumps({"<|rwkv_tokenizer_end_of_text|>": 0}),
        encoding="utf-8",
    )
    special_tokens_map = {
        "bos_token": "<|rwkv_tokenizer_end_of_text|>",
        "eos_token": "\n\n",
        "unk_token": "<|rwkv_tokenizer_end_of_text|>",
        "pad_token": "<|rwkv_tokenizer_end_of_text|>",
    }
    (path / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map),
        encoding="utf-8",
    )
    tokenizer_config = {
        "auto_map": {
            "AutoTokenizer": [
                "hf_rwkv_tokenizer.RwkvTokenizer",
                None,
            ]
        },
        "tokenizer_class": "RwkvTokenizer",
        "added_tokens_decoder": {
            "0": {
                "content": "<|rwkv_tokenizer_end_of_text|>",
                "special": True,
            }
        },
        **special_tokens_map,
        "chat_template": (
            "{{ '<|rwkv_tokenizer_end_of_text|>' }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{'User: ' + message['content'] + '\\n\\n'}}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
        ),
    }
    (path / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config),
        encoding="utf-8",
    )
    return vocab_path


def test_resolve_tokenizer_args_detects_rwkv_txt_vocab(tmp_path):
    vocab_path = tmp_path / "rwkv_vocab_v20250609.txt"
    _write_vocab(vocab_path)

    tokenizer_mode, tokenizer_name, _, _ = resolve_tokenizer_args(str(vocab_path))

    assert tokenizer_mode == "rwkv"
    assert str(tokenizer_name) == str(vocab_path)


def test_resolve_tokenizer_args_detects_hf_rwkv_tokenizer_dir(tmp_path):
    _write_hf_rwkv_tokenizer_dir(tmp_path)

    tokenizer_mode, tokenizer_name, _, _ = resolve_tokenizer_args(str(tmp_path))

    assert tokenizer_mode == "rwkv"
    assert str(tokenizer_name) == str(tmp_path)


def test_rwkv_tokenizer_round_trips_and_prefers_longest_match(tmp_path):
    vocab_path = tmp_path / "rwkv_vocab_v20250609.txt"
    _write_vocab(vocab_path)

    tokenizer = get_tokenizer(str(vocab_path))

    assert tokenizer.encode("ab") == [3]
    assert tokenizer.decode([3, 2]) == "abb"
    assert tokenizer.convert_ids_to_tokens([3, 2]) == ["ab", "b"]
    assert tokenizer.convert_ids_to_tokens([6, 3, 7], skip_special_tokens=True) == [
        "ab"
    ]
    assert tokenizer.convert_tokens_to_ids(["ab", "b"]) == [3, 2]
    assert tokenizer.convert_tokens_to_string(["ab", "b"]) == "abb"
    assert tokenizer.get_vocab()["ab"] == 3
    assert tokenizer.get_added_vocab() == {}
    assert tokenizer.eos_token_id == 5
    assert tokenizer.bos_token_id == 5
    assert tokenizer.pad_token_id == 5
    assert tokenizer.all_special_ids == [6, 7, 5]
    assert tokenizer.vocab_size == 64


def test_rwkv_tokenizer_preserves_hf_added_token_semantics(tmp_path):
    _write_hf_rwkv_tokenizer_dir(tmp_path)

    tokenizer = get_tokenizer(str(tmp_path))

    assert tokenizer.encode("a\n\nb") == [1, 10, 2]
    assert tokenizer.decode([0, 1, 10, 2]) == ("<|rwkv_tokenizer_end_of_text|>a\n\nb")
    assert tokenizer.convert_tokens_to_ids("\n\n") == 10
    assert tokenizer.convert_ids_to_tokens([0, 1, 10, 2]) == [
        "<|rwkv_tokenizer_end_of_text|>",
        "a",
        "\n\n",
        "b",
    ]
    assert tokenizer.get_added_vocab() == {
        "<|rwkv_tokenizer_end_of_text|>": 0,
        "\n\n": 10,
    }
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 10
    assert tokenizer.pad_token_id == 0
    assert tokenizer.all_special_ids == [0, 10]
    assert tokenizer.vocab_size == 64


def test_rwkv_tokenizer_prioritizes_hf_added_token_boundaries(tmp_path):
    vocab_path = _write_hf_rwkv_tokenizer_dir(tmp_path)
    with vocab_path.open("a", encoding="utf-8") as f:
        f.write("10 'a\\n' 2\n")
        f.write("11 '\\n' 1\n")

    tokenizer = get_tokenizer(str(tmp_path))

    assert tokenizer.encode("a\n\nb") == [1, 12, 2]
    assert tokenizer.convert_tokens_to_ids("\n\n") == 12
    assert tokenizer.decode([1, 12, 2]) == "a\n\nb"


def test_rwkv_tokenizer_uses_hf_chat_template_prefix(tmp_path):
    _write_hf_rwkv_tokenizer_dir(tmp_path)
    tokenizer = get_tokenizer(str(tmp_path))

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ab"}],
        add_generation_prompt=True,
    )

    assert rendered == "<|rwkv_tokenizer_end_of_text|>User: ab\n\nAssistant:"
    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "ab"}],
        tokenize=True,
    ) == [0, 4, 5, 6, 7, 8, 9, 1, 2, 10]
