# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast, overload

from transformers import BatchEncoding

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from .protocol import TokenizerLike


class _TrieNode:
    __slots__ = ("children", "values")

    def __init__(self) -> None:
        self.children: list[_TrieNode | None] = [None] * 256
        self.values: set[tuple[bytes, int]] = set()

    def add(self, key: bytes, token_id: int, idx: int = 0) -> None:
        if idx == len(key):
            self.values.add((key, token_id))
            return
        ch = key[idx]
        child = self.children[ch]
        if child is None:
            child = _TrieNode()
            self.children[ch] = child
        child.add(key, token_id, idx + 1)

    def find_longest(self, key: bytes, idx: int) -> tuple[int, set[tuple[bytes, int]]]:
        node = self
        ch = key[idx]
        best: tuple[int, set[tuple[bytes, int]]] | None = None
        while node.children[ch] is not None:
            node = cast(_TrieNode, node.children[ch])
            idx += 1
            if node.values:
                best = (idx, node.values)
            if idx == len(key):
                break
            ch = key[idx]
        if best is None:
            raise ValueError("Failed to match a token in RWKV vocabulary.")
        return best


class RWKVTokenizer(TokenizerLike):
    _CHAT_ROLE_PREFIX = {
        "system": "System: ",
        "user": "User: ",
        "assistant": "Assistant: ",
        "tool": "Tool: ",
    }

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> RWKVTokenizer:
        del args, trust_remote_code, revision, download_dir
        vocab_path = cls._resolve_vocab_path(path_or_repo_id)
        truncation_side = kwargs.pop("truncation_side", "left")
        return cls(vocab_path=vocab_path, truncation_side=truncation_side)

    @staticmethod
    def _resolve_vocab_path(path_or_repo_id: str | Path) -> Path:
        path = Path(path_or_repo_id)
        if path.is_file():
            return path
        if path.is_dir():
            candidates = sorted(path.glob("rwkv_vocab*.txt"))
            if not candidates:
                candidates = sorted(path.glob("*.txt"))
            if len(candidates) == 1:
                return candidates[0]
        raise ValueError(f"Unable to locate an RWKV vocab txt file under {path}.")

    def __init__(self, *, vocab_path: Path, truncation_side: str = "left") -> None:
        self.vocab_path = vocab_path
        self._truncation_side = truncation_side
        self._id_to_token: dict[int, bytes] = {}
        self._token_to_id: dict[bytes, int] = {}

        with vocab_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                token_id = int(line[: line.index(" ")])
                token = eval(line[line.index(" ") : line.rindex(" ")])
                token_bytes = token.encode("utf-8") if isinstance(token, str) else token
                assert isinstance(token_bytes, bytes)
                self._id_to_token[token_id] = token_bytes
                self._token_to_id[token_bytes] = token_id

        self._root = _TrieNode()
        for token_bytes, token_id in self._token_to_id.items():
            self._root.add(token_bytes, token_id)

        self._known_token_slots = max(self._id_to_token) + 1
        self._vocab_size = ((self._known_token_slots + 63) // 64) * 64
        self._id_to_token_str = {
            token_id: self._token_bytes_to_str(token_bytes)
            for token_id, token_bytes in self._id_to_token.items()
        }
        self._token_str_to_id = {
            token_str: token_id for token_id, token_str in self._id_to_token_str.items()
        }
        self._vocab = dict(self._token_str_to_id)
        self._empty_added_vocab: dict[str, int] = {}
        self._special_token_map = self._build_special_token_map()
        self._all_special_tokens = list(self._special_token_map)
        self._all_special_ids = [
            self._special_token_map[token] for token in self._all_special_tokens
        ]
        self._all_special_ids_set = set(self._all_special_ids)
        self._default_special_id = self._resolve_default_special_id()
        self._max_chars_per_token = max(
            len(token) for token in self._id_to_token.values()
        )

    def _build_special_token_map(self) -> dict[str, int]:
        special_tokens: dict[str, int] = {}
        for token in (
            b"<|im_start|>",
            b"<|im_end|>",
            b"<|endoftext|>",
        ):
            token_id = self._token_to_id.get(token)
            if token_id is not None:
                special_tokens[token.decode("utf-8")] = token_id
        return special_tokens

    def _resolve_default_special_id(self) -> int:
        for token in (b"<|endoftext|>", b"\n\n"):
            token_id = self._token_to_id.get(token)
            if token_id is not None:
                return token_id
        return 0

    @staticmethod
    def _token_bytes_to_str(token: bytes) -> str:
        return token.decode("latin-1")

    @staticmethod
    def _token_str_to_bytes(token: str) -> bytes:
        return token.encode("latin-1")

    def num_special_tokens_to_add(self) -> int:
        return 0

    @property
    def all_special_tokens(self) -> list[str]:
        return self._all_special_tokens

    @property
    def all_special_ids(self) -> list[int]:
        return self._all_special_ids

    @property
    def bos_token_id(self) -> int:
        return self._special_token_map.get("<|endoftext|>", self._default_special_id)

    @property
    def eos_token_id(self) -> int:
        return self._special_token_map.get("<|endoftext|>", self._default_special_id)

    @property
    def pad_token_id(self) -> int:
        return self._special_token_map.get("<|endoftext|>", self._default_special_id)

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def max_token_id(self) -> int:
        return self._vocab_size - 1

    @property
    def max_chars_per_token(self) -> int:
        return self._max_chars_per_token

    @property
    def truncation_side(self) -> str:
        return self._truncation_side

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> BatchEncoding:
        del text_pair, add_special_tokens
        if isinstance(text, str):
            input_ids = self.encode(text, truncation=truncation, max_length=max_length)
            return BatchEncoding(
                {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
            )

        batch_ids = [
            self.encode(item, truncation=truncation, max_length=max_length)
            for item in text
        ]
        return BatchEncoding(
            {
                "input_ids": batch_ids,
                "attention_mask": [[1] * len(ids) for ids in batch_ids],
            }
        )

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def get_added_vocab(self) -> dict[str, int]:
        return self._empty_added_vocab

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        del add_special_tokens
        src = text.encode("utf-8")
        idx = 0
        tokens: list[int] = []
        while idx < len(src):
            idx, values = self._root.find_longest(src, idx)
            _, token_id = next(iter(values))
            tokens.append(token_id)

        if truncation and max_length is not None and len(tokens) > max_length:
            if self._truncation_side == "left":
                tokens = tokens[-max_length:]
            else:
                tokens = tokens[:max_length]
        return tokens

    def apply_chat_template(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> str | list[int]:
        if tools:
            raise ValueError("RWKV txt tokenizer does not support tool schemas.")
        add_generation_prompt = kwargs.get("add_generation_prompt", False)
        tokenize = kwargs.get("tokenize", False)

        rendered_parts: list[str] = []
        for message in messages:
            role = cast(str, message["role"])
            prefix = self._CHAT_ROLE_PREFIX.get(role, f"{role.title()}: ")
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                content = "".join(text_parts)
            rendered_parts.append(f"{prefix}{content}\n\n")
        if add_generation_prompt:
            rendered_parts.append("Assistant:")
        rendered = "".join(rendered_parts)
        return self.encode(rendered) if tokenize else rendered

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self._token_str_to_id.get(tokens, self._default_special_id)
        return [self.convert_tokens_to_ids(token) for token in tokens]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        raw = b"".join(
            self._id_to_token.get(self._token_str_to_id.get(token, -1), b"")
            for token in tokens
        )
        return raw.decode("utf-8", errors="replace")

    def decode(
        self, ids: Sequence[int] | int, skip_special_tokens: bool = False
    ) -> str:
        if isinstance(ids, int):
            ids = [ids]
        token_bytes = []
        special_ids = self._all_special_ids_set if skip_special_tokens else ()
        for token_id in ids:
            if token_id in special_ids:
                continue
            token_bytes.append(self._id_to_token.get(token_id, b""))
        return b"".join(token_bytes).decode("utf-8", errors="replace")

    def convert_ids_to_tokens(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        special_ids = self._all_special_ids_set if skip_special_tokens else ()
        tokens = []
        for token_id in ids:
            if token_id in special_ids:
                continue
            tokens.append(self._id_to_token_str.get(token_id, ""))
        return tokens
