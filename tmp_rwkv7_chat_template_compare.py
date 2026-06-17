# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare RWKV7 chat rendering with and without ``chat_template.jinja``.

Run from the repo root with the vLLM virtualenv:

    .venv/bin/python tmp_rwkv7_chat_template_compare.py

What this prints:

1. The prompt rendered by ``chat_template.jinja`` and the new
   ``rwkv_vocab_v20260603.txt`` special-token handling.
2. The prompt rendered by the legacy RWKV fallback
   (``System:`` / ``User:`` / ``Assistant:``), using the same vocab.
3. Token counts, first token ids, special marker ids, and a unified diff.

The "without template" side intentionally points metadata loading at an empty
temporary directory. This prevents the sibling ``chat_template.jinja`` file
from being auto-loaded, so the comparison shows only the template effect.

Optional model generation comparison:

    .venv/bin/python tmp_rwkv7_chat_template_compare.py --run-llm

Useful variants:

    .venv/bin/python tmp_rwkv7_chat_template_compare.py --enable-thinking
    .venv/bin/python tmp_rwkv7_chat_template_compare.py --user "你好，介绍一下你自己"
"""

from __future__ import annotations

import argparse
import difflib
import os
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm.tokenizers.rwkv import RWKVTokenizer

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VOCAB = REPO_ROOT / "rwkv_vocab_v20260603.txt"
DEFAULT_TEMPLATE = REPO_ROOT / "chat_template.jinja"
DEFAULT_MODEL = Path("/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF")


def _build_messages(system: str, user: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": system,
            "current_date": "2026-06-03",
            "current_location": "Shanghai, China",
        },
        {
            "role": "user",
            "content": user,
        },
    ]


def _print_block(title: str, text: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(text)
    print("-" * 80)
    print("repr:")
    print(repr(text))


def _print_token_summary(tokenizer: RWKVTokenizer, text: str) -> None:
    ids = tokenizer.encode(text)
    print("token count:", len(ids))
    print("first 80 token ids:", ids[:80])
    print(
        "special ids:",
        {
            marker: tokenizer.convert_tokens_to_ids(marker)
            for marker in (
                "<|im_start|>",
                "<|im_end|>",
                "<|endoftext|>",
                "<|think|>",
                "<|tool_call|>",
            )
        },
    )


def _maybe_run_llm(args, prompts: list[tuple[str, str]]) -> None:
    if not args.run_llm:
        return

    print("\n" + "=" * 80)
    print("MODEL GENERATION")
    print("=" * 80)
    print("model:", args.model)
    print("max_tokens:", args.max_tokens)
    print("temperature:", args.temperature)

    os.environ.setdefault("VLLM_ENGINE_ITERATION_TIMEOUT_S", "120")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.vocab),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        enable_prefix_caching=False,
    )
    outputs = llm.generate(
        [prompt for _, prompt in prompts],
        SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature),
    )
    for (name, prompt), request_output in zip(prompts, outputs):
        print("\n" + "-" * 80)
        print(name)
        print("-" * 80)
        print("prompt tail:")
        print(prompt[-500:])
        print("\ngenerated text:")
        print(request_output.outputs[0].text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RWKV7 chat rendering with chat_template.jinja against "
            "the legacy role-prefix fallback."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              .venv/bin/python tmp_rwkv7_chat_template_compare.py
              .venv/bin/python tmp_rwkv7_chat_template_compare.py --enable-thinking
              .venv/bin/python tmp_rwkv7_chat_template_compare.py --run-llm
              .venv/bin/python tmp_rwkv7_chat_template_compare.py --user "你好"

            Notes:
              The default mode only renders and tokenizes prompts. It does not
              load the model. Use --run-llm when you want to compare generated
              text from the two rendered prompts.
            """
        ),
    )
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--system",
        default="你是小科，一个有帮助的中文助手。回答要简洁。",
    )
    parser.add_argument(
        "--user",
        default="9 + 3 - 4 是多少？请直接给数字。",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=True into chat_template.jinja.",
    )
    parser.add_argument(
        "--run-llm",
        action="store_true",
        help="Also run vLLM.generate on both rendered prompts and print outputs.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.40)
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Allow non-eager execution when running the optional LLM path.",
    )
    parser.set_defaults(enforce_eager=True)
    args = parser.parse_args()

    messages = _build_messages(args.system, args.user)
    template_text = args.template.read_text(encoding="utf-8")

    with_template = RWKVTokenizer.from_pretrained(
        args.vocab,
        chat_template=template_text,
    )

    # Empty metadata dir disables auto-loading tokenizer_config/chat_template
    # while keeping the same vocab file for a fair A/B render comparison.
    with TemporaryDirectory() as empty_metadata_dir:
        without_template = RWKVTokenizer.from_pretrained(
            args.vocab,
            metadata_dir=empty_metadata_dir,
        )

        common_kwargs = {
            "add_generation_prompt": True,
            "enable_thinking": args.enable_thinking,
        }
        rendered_with = with_template.apply_chat_template(messages, **common_kwargs)
        rendered_without = without_template.apply_chat_template(
            messages,
            **common_kwargs,
        )

    assert isinstance(rendered_with, str)
    assert isinstance(rendered_without, str)

    print("vocab:", args.vocab)
    print("template:", args.template)
    print("enable_thinking:", args.enable_thinking)
    print("run_llm:", args.run_llm)
    print("messages:", messages)

    _print_block("WITH chat_template.jinja", rendered_with)
    _print_token_summary(with_template, rendered_with)

    _print_block("WITHOUT chat_template.jinja (legacy fallback)", rendered_without)
    _print_token_summary(without_template, rendered_without)

    print("\n" + "=" * 80)
    print("UNIFIED DIFF: with template vs without template")
    print("=" * 80)
    diff = difflib.unified_diff(
        rendered_without.splitlines(keepends=True),
        rendered_with.splitlines(keepends=True),
        fromfile="without_chat_template",
        tofile="with_chat_template",
    )
    print("".join(diff))

    _maybe_run_llm(
        args,
        [
            ("WITHOUT chat_template.jinja (legacy fallback)", rendered_without),
            ("WITH chat_template.jinja", rendered_with),
        ],
    )


if __name__ == "__main__":
    main()
