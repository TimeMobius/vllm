#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from tmp_rwkv7_pure_torch_generate import (
    NativeRWKV7,
    accelerator_available,
    analyze_generation_case,
    clear_device_cache,
    infer_checkpoint_summary,
    load_checkpoint,
    load_tokenizer,
    prepare_prompt_inputs,
    resolve_dtype,
    run_prompt,
)

Role = Literal["system", "user", "assistant", "tool"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve the standalone native RWKV7 generator over FastAPI. The model "
            "is loaded once at startup and reused across requests."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--device", default="cuda" if accelerator_available() else "cpu"
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-max-new-tokens", type=int, default=64)
    parser.add_argument("--default-topk", type=int, default=8)
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Use the tokenizer chat template by default for generation requests.",
    )
    parser.add_argument(
        "--message-role",
        default="user",
        choices=("system", "user", "assistant", "tool"),
    )
    parser.add_argument("--system-prompt")
    parser.add_argument(
        "--no-add-generation-prompt",
        action="store_true",
        help="Disable add_generation_prompt in the default request config.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
    )
    return parser.parse_args()


class PrepareRequest(BaseModel):
    prompt: str = Field(..., description="Raw user prompt to prepare.")
    apply_chat_template: bool | None = None
    message_role: Role | None = None
    system_prompt: str | None = None
    add_generation_prompt: bool | None = None


class GenerateRequest(PrepareRequest):
    max_new_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Maximum number of new tokens to greedily generate.",
    )
    topk: int | None = Field(
        default=None,
        ge=1,
        description="Number of first-step top-k logits to return.",
    )
    include_analysis: bool = Field(
        default=True,
        description="Include the degeneracy analysis block in the response.",
    )


class RWKVService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_path = args.model
        self.tokenizer_path = args.tokenizer
        self.device = torch.device(args.device)
        self.dtype = resolve_dtype(args.dtype)
        self.default_max_new_tokens = args.default_max_new_tokens
        self.default_topk = args.default_topk
        self.default_apply_chat_template = args.apply_chat_template
        self.default_message_role: Role = args.message_role
        self.default_system_prompt = args.system_prompt
        self.default_add_generation_prompt = not args.no_add_generation_prompt

        self.tokenizer = load_tokenizer(
            args.tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
        checkpoint = load_checkpoint(Path(args.model))
        checkpoint_summary = infer_checkpoint_summary(checkpoint)
        self.model = NativeRWKV7(
            checkpoint,
            device=self.device,
            dtype=self.dtype,
        )
        self.summary = {
            "model": self.model_path,
            "tokenizer": self.tokenizer_path,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "vocab_size": checkpoint_summary.vocab_size,
            "hidden_size": checkpoint_summary.hidden_size,
            "num_layers": checkpoint_summary.num_layers,
            "tokenizer_full_vocab_size": len(self.tokenizer),
            "special_ids": {
                "bos_token_id": getattr(self.tokenizer, "bos_token_id", None),
                "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
                "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            },
            "default_request_config": {
                "max_new_tokens": self.default_max_new_tokens,
                "topk": self.default_topk,
                "apply_chat_template": self.default_apply_chat_template,
                "message_role": self.default_message_role,
                "system_prompt": self.default_system_prompt,
                "add_generation_prompt": self.default_add_generation_prompt,
            },
        }
        del checkpoint
        clear_device_cache()

    def close(self) -> None:
        self.model = None
        clear_device_cache()

    def resolve_request_config(
        self,
        request: PrepareRequest | GenerateRequest,
    ) -> dict[str, object]:
        return {
            "apply_chat_template": (
                self.default_apply_chat_template
                if request.apply_chat_template is None
                else request.apply_chat_template
            ),
            "message_role": (
                self.default_message_role
                if request.message_role is None
                else request.message_role
            ),
            "system_prompt": (
                self.default_system_prompt
                if request.system_prompt is None
                else request.system_prompt
            ),
            "add_generation_prompt": (
                self.default_add_generation_prompt
                if request.add_generation_prompt is None
                else request.add_generation_prompt
            ),
        }

    def prepare(self, request: PrepareRequest) -> dict[str, object]:
        config = self.resolve_request_config(request)
        prepared = prepare_prompt_inputs(
            self.tokenizer,
            request.prompt,
            apply_chat_template=bool(config["apply_chat_template"]),
            message_role=str(config["message_role"]),
            system_prompt=(
                None
                if config["system_prompt"] is None
                else str(config["system_prompt"])
            ),
            add_generation_prompt=bool(config["add_generation_prompt"]),
        )
        prepared["effective_request_config"] = config
        return prepared

    def generate(self, request: GenerateRequest) -> dict[str, object]:
        config = self.resolve_request_config(request)
        result = run_prompt(
            self.model,
            self.tokenizer,
            request.prompt,
            max_new_tokens=(
                self.default_max_new_tokens
                if request.max_new_tokens is None
                else request.max_new_tokens
            ),
            topk=self.default_topk if request.topk is None else request.topk,
            apply_chat_template=bool(config["apply_chat_template"]),
            message_role=str(config["message_role"]),
            system_prompt=(
                None
                if config["system_prompt"] is None
                else str(config["system_prompt"])
            ),
            add_generation_prompt=bool(config["add_generation_prompt"]),
        )
        result["effective_request_config"] = {
            **config,
            "max_new_tokens": (
                self.default_max_new_tokens
                if request.max_new_tokens is None
                else request.max_new_tokens
            ),
            "topk": self.default_topk if request.topk is None else request.topk,
        }
        if request.include_analysis:
            result["analysis"] = analyze_generation_case(result)
        return result


def create_app(args: argparse.Namespace) -> FastAPI:
    service = RWKVService(args)
    app = FastAPI(
        title="RWKV7 Pure Torch FastAPI Service",
        version="0.1.0",
        summary="Standalone native RWKV7 generation service without the vLLM engine.",
    )
    app.state.service = service
    app.state.generate_lock = asyncio.Lock()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        app.state.service.close()

    @app.get("/")
    async def root() -> dict[str, object]:
        return {
            "service": "rwkv7-pure-torch-fastapi",
            "status": "ok",
            "endpoints": {
                "health": "/health",
                "summary": "/summary",
                "prepare": "/prepare",
                "generate": "/generate",
                "docs": "/docs",
            },
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/summary")
    async def summary() -> dict[str, object]:
        return app.state.service.summary

    @app.post("/prepare")
    async def prepare(request: PrepareRequest) -> dict[str, object]:
        try:
            return await run_in_threadpool(app.state.service.prepare, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/generate")
    async def generate(request: GenerateRequest) -> dict[str, object]:
        async with app.state.generate_lock:
            try:
                return await run_in_threadpool(app.state.service.generate, request)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
