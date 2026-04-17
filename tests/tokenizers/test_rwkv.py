# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


def test_resolve_tokenizer_args_detects_rwkv_txt_vocab(tmp_path):
    vocab_path = tmp_path / "rwkv_vocab_v20250609.txt"
    _write_vocab(vocab_path)

    tokenizer_mode, tokenizer_name, _, _ = resolve_tokenizer_args(str(vocab_path))

    assert tokenizer_mode == "rwkv"
    assert str(tokenizer_name) == str(vocab_path)


def test_rwkv_tokenizer_round_trips_and_prefers_longest_match(tmp_path):
    vocab_path = tmp_path / "rwkv_vocab_v20250609.txt"
    _write_vocab(vocab_path)

    tokenizer = get_tokenizer(str(vocab_path))

    assert tokenizer.encode("ab") == [3]
    assert tokenizer.decode([3, 2]) == "abb"
    assert tokenizer.eos_token_id == 5
    assert tokenizer.bos_token_id == 5
    assert tokenizer.pad_token_id == 5
    assert tokenizer.all_special_ids == [6, 7, 5]
    assert tokenizer.vocab_size == 64
