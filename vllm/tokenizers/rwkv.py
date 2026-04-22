# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from ast import literal_eval
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast, overload

import regex as re
from transformers import BatchEncoding

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from .protocol import TokenizerLike

try:
    from pyrwkv_tokenizer import WorldTokenizer as FastWorldTokenizer
except ImportError:
    FastWorldTokenizer = None


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
    _SPECIAL_TOKEN_KEYS = (
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
        "sep_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    )

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
        root_path = Path(path_or_repo_id)
        vocab_path = cls._resolve_vocab_path(root_path)
        truncation_side = kwargs.pop("truncation_side", "left")
        metadata = (
            cls._load_tokenizer_metadata(root_path)
            if root_path.is_dir()
            else cls._load_tokenizer_metadata(vocab_path.parent)
        )
        return cls(
            vocab_path=vocab_path,
            truncation_side=truncation_side,
            metadata=metadata,
        )

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

    @staticmethod
    def _load_json_file(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError, TypeError):
            return {}

        return data if isinstance(data, dict) else {}

    @classmethod
    def _normalize_special_token(cls, value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            content = value.get("content")
            if isinstance(content, str):
                return content
        return None

    @classmethod
    def _load_tokenizer_metadata(cls, path: Path) -> dict[str, Any]:
        tokenizer_config = cls._load_json_file(path / "tokenizer_config.json")
        special_tokens_map = cls._load_json_file(path / "special_tokens_map.json")
        added_tokens = cls._load_json_file(path / "added_tokens.json")

        explicit_special_token_ids: dict[str, int] = {}
        for added_token, token_id in added_tokens.items():
            if isinstance(added_token, str) and isinstance(token_id, int):
                explicit_special_token_ids[added_token] = token_id

        added_tokens_decoder = tokenizer_config.get("added_tokens_decoder", {})
        if isinstance(added_tokens_decoder, dict):
            for token_id, token_spec in added_tokens_decoder.items():
                decoded_token = cls._normalize_special_token(token_spec)
                if decoded_token is None:
                    continue
                try:
                    explicit_special_token_ids.setdefault(decoded_token, int(token_id))
                except (TypeError, ValueError):
                    continue

        ordered_special_tokens: list[str] = []
        special_token_fields: dict[str, str] = {}
        for source in (tokenizer_config, special_tokens_map):
            for key in cls._SPECIAL_TOKEN_KEYS:
                value = source.get(key)
                if key == "additional_special_tokens" and isinstance(value, list):
                    for item in value:
                        extra_token = cls._normalize_special_token(item)
                        if (
                            extra_token is not None
                            and extra_token not in ordered_special_tokens
                        ):
                            ordered_special_tokens.append(extra_token)
                    continue

                field_token = cls._normalize_special_token(value)
                if field_token is None:
                    continue
                if field_token not in ordered_special_tokens:
                    ordered_special_tokens.append(field_token)
                special_token_fields.setdefault(key, field_token)

        return {
            "ordered_special_tokens": ordered_special_tokens,
            "explicit_special_token_ids": explicit_special_token_ids,
            "special_token_fields": special_token_fields,
            "chat_template": tokenizer_config.get("chat_template"),
        }

    def __init__(
        self,
        *,
        vocab_path: Path,
        truncation_side: str = "left",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.vocab_path = vocab_path
        self._truncation_side = truncation_side
        self._id_to_token: dict[int, bytes] = {}
        self._token_to_id: dict[bytes, int] = {}

        with vocab_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                token_id = int(line[: line.index(" ")])
                token = literal_eval(line[line.index(" ") : line.rindex(" ")])
                token_bytes = token.encode("utf-8") if isinstance(token, str) else token
                assert isinstance(token_bytes, bytes)
                self._id_to_token[token_id] = token_bytes
                self._token_to_id[token_bytes] = token_id

        self._root = _TrieNode()
        for token_bytes, token_id in self._token_to_id.items():
            self._root.add(token_bytes, token_id)

        self._id_to_token_str = {
            token_id: self._token_bytes_to_str(token_bytes)
            for token_id, token_bytes in self._id_to_token.items()
        }
        self._token_str_to_id = {
            token_str: token_id for token_id, token_str in self._id_to_token_str.items()
        }
        self._special_token_map, special_fields = self._build_special_token_map(
            metadata or {}
        )
        self._special_id_to_token = {
            token_id: token for token, token_id in self._special_token_map.items()
        }
        self._all_special_tokens = list(self._special_token_map)
        self._all_special_ids = [
            self._special_token_map[token] for token in self._all_special_tokens
        ]
        self._all_special_ids_set = set(self._all_special_ids)
        self._added_vocab = {
            token: token_id
            for token, token_id in self._special_token_map.items()
            if self._token_str_to_id.get(token) != token_id
        }
        self._vocab = {**self._token_str_to_id, **self._added_vocab}
        self._fast_backend_supports_added_vocab = False
        self._fast_backend_vocab_size: int | None = None
        self._fast_backend = self._build_fast_backend()
        max_token_id = max(
            [*self._id_to_token.keys(), *self._special_token_map.values()],
            default=0,
        )
        self._known_token_slots = max_token_id + 1
        self._vocab_size = ((self._known_token_slots + 63) // 64) * 64
        self._default_special_id = self._resolve_default_special_id()
        self._bos_token_id = self._resolve_named_special_id(
            special_fields.get("bos_token")
        )
        self._eos_token_id = self._resolve_named_special_id(
            special_fields.get("eos_token")
        )
        self._pad_token_id = self._resolve_named_special_id(
            special_fields.get("pad_token")
        )
        self._max_chars_per_token = max(
            [
                *(len(token) for token in self._id_to_token.values()),
                *(len(token.encode("utf-8")) for token in self._all_special_tokens),
            ],
            default=0,
        )
        self._special_token_pattern = self._build_special_token_pattern()
        chat_template = metadata.get("chat_template") if metadata else None
        bos_token = special_fields.get("bos_token")
        self._chat_prefix = (
            bos_token
            if isinstance(chat_template, str)
            and bos_token is not None
            and bos_token in chat_template
            else ""
        )

    def _build_fast_backend(self) -> Any | None:
        if FastWorldTokenizer is None:
            return None

        from_buffer = getattr(FastWorldTokenizer, "from_buffer", None)
        if self._added_vocab and callable(from_buffer):
            max_rust_token_id = (1 << 16) - 1
            if all(
                0 <= token_id <= max_rust_token_id
                for token_id in self._added_vocab.values()
            ):
                try:
                    backend = from_buffer(self._build_augmented_vocab_buffer())
                except Exception:
                    backend = None
                else:
                    self._fast_backend_supports_added_vocab = True
                    self._fast_backend_vocab_size = self._get_fast_backend_vocab_size(
                        backend
                    )
                    return backend

        try:
            backend = FastWorldTokenizer(str(self.vocab_path))
        except Exception:
            return None
        self._fast_backend_vocab_size = self._get_fast_backend_vocab_size(backend)
        return backend

    @staticmethod
    def _get_fast_backend_vocab_size(backend: Any) -> int | None:
        vocab_size = getattr(backend, "vocab_size", None)
        if not callable(vocab_size):
            return None
        try:
            return int(vocab_size())
        except Exception:
            return None

    def _build_augmented_vocab_buffer(self) -> bytes:
        buffer = self.vocab_path.read_bytes()
        if buffer and not buffer.endswith(b"\n"):
            buffer += b"\n"
        for token, token_id in sorted(
            self._added_vocab.items(), key=lambda item: item[1]
        ):
            token_bytes = token.encode("utf-8")
            buffer += f"{token_id} {token!r} {len(token_bytes)}\n".encode()
        return buffer

    def _build_special_token_map(
        self, metadata: dict[str, Any]
    ) -> tuple[dict[str, int], dict[str, str]]:
        ordered_special_tokens = list(
            dict.fromkeys(
                [
                    *metadata.get("ordered_special_tokens", []),
                    *metadata.get("explicit_special_token_ids", {}).keys(),
                ]
            )
        )
        explicit_special_token_ids = metadata.get("explicit_special_token_ids", {})
        special_fields = metadata.get("special_token_fields", {})

        if ordered_special_tokens:
            next_token_id = (
                max(
                    [*self._id_to_token.keys(), *explicit_special_token_ids.values()],
                    default=-1,
                )
                + 1
            )
            special_tokens: dict[str, int] = {}
            for token in ordered_special_tokens:
                token_id = explicit_special_token_ids.get(token)
                if token_id is None:
                    token_id = next_token_id
                    next_token_id += 1
                special_tokens[token] = token_id
            return special_tokens, special_fields

        default_special_tokens: dict[str, int] = {}
        for token_bytes in (
            b"<|im_start|>",
            b"<|im_end|>",
            b"<|endoftext|>",
        ):
            token_id = self._token_to_id.get(token_bytes)
            if token_id is not None:
                default_special_tokens[token_bytes.decode("utf-8")] = token_id
        return default_special_tokens, {}

    def _resolve_named_special_id(self, token: str | None) -> int:
        if token is None:
            return self._default_special_id
        return self._special_token_map.get(token, self._default_special_id)

    def _resolve_default_special_id(self) -> int:
        for token in (
            "<|rwkv_tokenizer_end_of_text|>",
            "<|endoftext|>",
            "\n\n",
        ):
            token_id = self._special_token_map.get(token)
            if token_id is not None:
                return token_id
        for token_bytes in (b"<|endoftext|>", b"\n\n"):
            token_id = self._token_to_id.get(token_bytes)
            if token_id is not None:
                return token_id
        return 0

    def _build_special_token_pattern(self) -> re.Pattern[str] | None:
        if not self._special_token_map:
            return None

        tokens = sorted(self._special_token_map, key=len, reverse=True)
        return re.compile("|".join(re.escape(token) for token in tokens))

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
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def is_fast(self) -> bool:
        return self._fast_backend is not None

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

        batch_ids = self._encode_batch(
            text,
            truncation=truncation,
            max_length=max_length,
        )
        return BatchEncoding(
            {
                "input_ids": batch_ids,
                "attention_mask": [[1] * len(ids) for ids in batch_ids],
            }
        )

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added_vocab)

    def _split_special_tokens(self, text: str) -> list[tuple[bool, str]]:
        if self._special_token_pattern is None:
            return [(False, text)] if text else []

        segments: list[tuple[bool, str]] = []
        last_idx = 0
        for match in self._special_token_pattern.finditer(text):
            if match.start() > last_idx:
                segments.append((False, text[last_idx : match.start()]))
            segments.append((True, match.group(0)))
            last_idx = match.end()

        if last_idx < len(text):
            segments.append((False, text[last_idx:]))

        return segments

    def _encode_plain_text(self, text: str) -> list[int]:
        if not text:
            return []
        if self._fast_backend is not None:
            return list(self._fast_backend.encode(text))

        src = text.encode("utf-8")
        idx = 0
        tokens: list[int] = []
        while idx < len(src):
            idx, values = self._root.find_longest(src, idx)
            _, token_id = next(iter(values))
            tokens.append(token_id)
        return tokens

    def _encode_plain_text_batch(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        if self._fast_backend is not None:
            return [list(ids) for ids in self._fast_backend.encode_batch(texts)]
        return [self._encode_plain_text(text) for text in texts]

    def _apply_truncation(
        self,
        tokens: list[int],
        truncation: bool | None,
        max_length: int | None,
    ) -> list[int]:
        if truncation and max_length is not None and len(tokens) > max_length:
            if self._truncation_side == "left":
                return tokens[-max_length:]
            return tokens[:max_length]
        return tokens

    def _encode_segments(
        self,
        segments: list[tuple[bool, str]],
        *,
        truncation: bool | None,
        max_length: int | None,
    ) -> list[int]:
        tokens: list[int] = []
        for is_special, segment in segments:
            if is_special:
                tokens.append(self._special_token_map[segment])
            else:
                tokens.extend(self._encode_plain_text(segment))
        return self._apply_truncation(tokens, truncation, max_length)

    def _encode_batch(
        self,
        texts: list[str],
        *,
        truncation: bool | None,
        max_length: int | None,
    ) -> list[list[int]]:
        split_texts = [self._split_special_tokens(text) for text in texts]
        plain_segments = [
            segment
            for split_text in split_texts
            for is_special, segment in split_text
            if not is_special and segment
        ]
        encoded_plain_segments = iter(self._encode_plain_text_batch(plain_segments))

        batch_ids: list[list[int]] = []
        for split_text in split_texts:
            tokens: list[int] = []
            for is_special, segment in split_text:
                if is_special:
                    tokens.append(self._special_token_map[segment])
                elif segment:
                    tokens.extend(next(encoded_plain_segments))
            batch_ids.append(self._apply_truncation(tokens, truncation, max_length))
        return batch_ids

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        del add_special_tokens
        return self._encode_segments(
            self._split_special_tokens(text),
            truncation=truncation,
            max_length=max_length,
        )

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

        rendered_parts: list[str] = [self._chat_prefix] if self._chat_prefix else []
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
            if tokens in self._special_token_map:
                return self._special_token_map[tokens]
            return self._token_str_to_id.get(tokens, self._default_special_id)
        return [self.convert_tokens_to_ids(token) for token in tokens]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.decode(cast(list[int], self.convert_tokens_to_ids(tokens)))

    def decode(
        self, ids: Sequence[int] | int, skip_special_tokens: bool = False
    ) -> str:
        if isinstance(ids, int):
            ids = [ids]
        ids = list(ids)
        if skip_special_tokens:
            ids = [
                token_id
                for token_id in ids
                if token_id not in self._all_special_ids_set
            ]
        fast_decoded = self._decode_with_fast_backend(ids)
        if fast_decoded is not None:
            return fast_decoded

        token_bytes = []
        for token_id in ids:
            if token_id in self._id_to_token:
                token_bytes.append(self._id_to_token[token_id])
            else:
                token = self._special_id_to_token.get(token_id)
                if token is not None:
                    token_bytes.append(token.encode("utf-8"))
        return b"".join(token_bytes).decode("utf-8", errors="replace")

    def _decode_with_fast_backend(self, ids: list[int]) -> str | None:
        if self._fast_backend is None:
            return None
        if self._added_vocab and not self._fast_backend_supports_added_vocab:
            return None
        if self._fast_backend_vocab_size is not None and any(
            token_id < 0 or token_id >= self._fast_backend_vocab_size
            for token_id in ids
        ):
            return None
        try:
            return str(self._fast_backend.decode(ids))
        except Exception:
            return None

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
            if token_id in self._id_to_token_str:
                tokens.append(self._id_to_token_str[token_id])
            else:
                tokens.append(self._special_id_to_token.get(token_id, ""))
        return tokens
