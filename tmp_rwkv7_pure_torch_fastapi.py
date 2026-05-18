#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

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
OpenAIRole = Literal["system", "developer", "user", "assistant", "tool"]


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
    parser.add_argument(
        "--served-model-name",
        help=(
            "Public model name exposed through the OpenAI-compatible /v1/* "
            "endpoints. Defaults to the checkpoint filename."
        ),
    )
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


class OpenAIContentPart(BaseModel):
    type: str
    text: str | None = None


class OpenAIChatMessage(BaseModel):
    role: OpenAIRole
    content: str | list[OpenAIContentPart] | None = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[OpenAIChatMessage]
    max_tokens: int | None = Field(default=None, ge=0)
    max_completion_tokens: int | None = Field(default=None, ge=0)
    temperature: float | None = None
    top_p: float | None = None
    n: int = Field(default=1, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None


class OpenAICompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int | None = Field(default=None, ge=0)
    temperature: float | None = None
    top_p: float | None = None
    n: int = Field(default=1, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None


def normalize_openai_role(role: OpenAIRole) -> Role:
    if role == "developer":
        return "system"
    return role


def flatten_message_content(
    content: str | list[OpenAIContentPart] | None,
) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for item in content:
        if item.type == "text" and item.text is not None:
            parts.append(item.text)
    return "".join(parts)


def normalize_openai_messages(
    messages: list[OpenAIChatMessage],
) -> list[dict[str, str]]:
    return [
        {
            "role": normalize_openai_role(message.role),
            "content": flatten_message_content(message.content),
        }
        for message in messages
    ]


def render_fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        label = message["role"].capitalize()
        chunks.append(f"{label}: {message['content']}")

    if not messages or messages[-1]["role"] != "assistant":
        chunks.append("Assistant:")
    return "\n\n".join(chunks)


def trim_stop_sequences(text: str, stop: str | list[str] | None) -> str:
    if stop is None:
        return text

    stop_values = [stop] if isinstance(stop, str) else stop
    cut_positions = [
        text.find(stop_value)
        for stop_value in stop_values
        if stop_value and text.find(stop_value) >= 0
    ]
    if not cut_positions:
        return text
    return text[: min(cut_positions)]


class RWKVService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_path = args.model
        self.tokenizer_path = args.tokenizer
        self.served_model_name = args.served_model_name or Path(args.model).name
        self.device = torch.device(args.device)
        self.dtype = resolve_dtype(args.dtype)
        self.default_max_new_tokens = args.default_max_new_tokens
        self.default_topk = args.default_topk
        self.default_apply_chat_template = args.apply_chat_template
        self.default_message_role: Role = args.message_role
        self.default_system_prompt = args.system_prompt
        self.default_add_generation_prompt = not args.no_add_generation_prompt
        self.accepted_model_names = {
            self.served_model_name,
            self.model_path,
            Path(self.model_path).name,
            Path(self.model_path).stem,
        }

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
            "served_model_name": self.served_model_name,
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

    def validate_requested_model(self, requested_model: str | None) -> str:
        if requested_model is None:
            return self.served_model_name
        if requested_model not in self.accepted_model_names:
            raise ValueError(
                f"Requested model {requested_model!r} does not match the loaded "
                f"model {self.served_model_name!r}."
            )
        return self.served_model_name

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

    def generate_from_rendered_prompt(
        self,
        rendered_prompt: str,
        *,
        max_new_tokens: int,
        topk: int,
        include_analysis: bool,
    ) -> dict[str, object]:
        result = run_prompt(
            self.model,
            self.tokenizer,
            rendered_prompt,
            max_new_tokens=max_new_tokens,
            topk=topk,
            apply_chat_template=False,
            message_role="user",
            system_prompt=None,
            add_generation_prompt=False,
        )
        if include_analysis:
            result["analysis"] = analyze_generation_case(result)
        return result

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

    def render_chat_messages(
        self,
        messages: list[OpenAIChatMessage],
    ) -> tuple[list[dict[str, str]], str]:
        normalized_messages = normalize_openai_messages(messages)
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                rendered_prompt = self.tokenizer.apply_chat_template(
                    normalized_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return normalized_messages, str(rendered_prompt)
            except Exception:
                pass

        return normalized_messages, render_fallback_chat_prompt(normalized_messages)


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
                "models": "/v1/models",
                "chat_completions": "/v1/chat/completions",
                "completions": "/v1/completions",
                "docs": "/docs",
            },
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/summary")
    async def summary() -> dict[str, object]:
        return app.state.service.summary

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        service = app.state.service
        return {
            "object": "list",
            "data": [
                {
                    "id": service.served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "rwkv",
                }
            ],
        }

    @app.get("/v1/models/{model_name}")
    async def retrieve_model(model_name: str) -> dict[str, object]:
        service = app.state.service
        try:
            resolved_name = service.validate_requested_model(model_name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return {
            "id": resolved_name,
            "object": "model",
            "created": 0,
            "owned_by": "rwkv",
        }

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

    @app.post("/v1/completions")
    async def openai_completions(
        request: OpenAICompletionRequest,
    ) -> dict[str, object] | StreamingResponse:
        service = app.state.service
        try:
            model_name = service.validate_requested_model(request.model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if request.n != 1:
            raise HTTPException(
                status_code=400,
                detail="Only n=1 is supported by this RWKV service.",
            )

        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        max_new_tokens = (
            service.default_max_new_tokens
            if request.max_tokens is None
            else request.max_tokens
        )
        topk = service.default_topk

        async with app.state.generate_lock:
            try:
                result = await run_in_threadpool(
                    service.generate_from_rendered_prompt,
                    request.prompt,
                    max_new_tokens=max_new_tokens,
                    topk=topk,
                    include_analysis=False,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        output_text = trim_stop_sequences(result["generated_text"], request.stop)
        finish_reason = (
            "length" if len(result["generated_ids"]) >= max_new_tokens else "stop"
        )
        usage = {
            "prompt_tokens": len(result["prompt_ids"]),
            "completion_tokens": len(result["generated_ids"]),
            "total_tokens": len(result["prompt_ids"]) + len(result["generated_ids"]),
        }

        if request.stream:

            async def completion_stream() -> Any:
                first_chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": output_text,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                final_chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                completion_stream(),
                media_type="text/event-stream",
            )

        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": output_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(
        request: OpenAIChatCompletionRequest,
    ) -> dict[str, object] | StreamingResponse:
        service = app.state.service
        try:
            model_name = service.validate_requested_model(request.model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if request.n != 1:
            raise HTTPException(
                status_code=400,
                detail="Only n=1 is supported by this RWKV service.",
            )

        normalized_messages, rendered_prompt = service.render_chat_messages(
            request.messages
        )
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        max_new_tokens = (
            service.default_max_new_tokens
            if request.max_completion_tokens is None and request.max_tokens is None
            else (
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens
            )
        )
        topk = service.default_topk

        async with app.state.generate_lock:
            try:
                result = await run_in_threadpool(
                    service.generate_from_rendered_prompt,
                    rendered_prompt,
                    max_new_tokens=max_new_tokens,
                    topk=topk,
                    include_analysis=False,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        output_text = trim_stop_sequences(result["generated_text"], request.stop)
        finish_reason = (
            "length" if len(result["generated_ids"]) >= max_new_tokens else "stop"
        )
        usage = {
            "prompt_tokens": len(result["prompt_ids"]),
            "completion_tokens": len(result["generated_ids"]),
            "total_tokens": len(result["prompt_ids"]) + len(result["generated_ids"]),
        }

        if request.stream:

            async def chat_stream() -> Any:
                role_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                content_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": output_text},
                            "finish_reason": None,
                        }
                    ],
                }
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                chat_stream(),
                media_type="text/event-stream",
            )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
            "rendered_prompt": rendered_prompt,
            "normalized_messages": normalized_messages,
        }

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
