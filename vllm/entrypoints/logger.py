# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

import torch

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import BeamSearchParams, SamplingParams

logger = init_logger(__name__)


class RequestLogger:
    def __init__(
        self,
        *,
        max_log_len: int | None,
        io_log_path: str | None = None,
        enable_log_requests: bool = True,
        enable_log_outputs: bool = True,
    ) -> None:
        self.max_log_len = max_log_len
        self.io_log_path = Path(io_log_path) if io_log_path else None
        self.enable_log_requests = enable_log_requests
        self.enable_log_outputs = enable_log_outputs
        self._io_log_lock = Lock()

        if self.io_log_path is not None:
            self.io_log_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.enable_log_requests and not self.enable_log_outputs:
            return

        if not logger.isEnabledFor(logging.INFO):
            logger.warning_once(
                "Request logging is enabled but "
                "the minimum log level is higher than INFO. "
                "No request information will be logged."
            )
        elif not logger.isEnabledFor(logging.DEBUG):
            logger.info_once(
                "Request logging is enabled but "
                "the minimum log level is higher than DEBUG. "
                "Only limited information will be logged to minimize overhead. "
                "To view more details, set `VLLM_LOGGING_LEVEL=DEBUG`."
            )

    def _write_io_log(self, event: str, payload: dict) -> None:
        if self.io_log_path is None:
            return

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **payload,
        }
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._io_log_lock:
                with self.io_log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(line)
                    log_file.write("\n")
        except Exception:
            logger.exception("Failed to write request IO log.")

    def log_inputs(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        params: SamplingParams | PoolingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        prompt_for_log = prompt
        prompt_token_ids_for_log = prompt_token_ids
        if logger.isEnabledFor(logging.DEBUG) and (
            max_log_len := self.max_log_len
        ) is not None:
            if prompt_for_log is not None:
                prompt_for_log = prompt_for_log[:max_log_len]

            if prompt_token_ids_for_log is not None:
                prompt_token_ids_for_log = prompt_token_ids_for_log[:max_log_len]

        if self.enable_log_requests and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Request %s details: prompt: %r, "
                "prompt_token_ids: %s, "
                "prompt_embeds shape: %s.",
                request_id,
                prompt_for_log,
                prompt_token_ids_for_log,
                prompt_embeds.shape if prompt_embeds is not None else None,
            )

        if self.enable_log_requests:
            logger.info(
                "Received request %s: params: %s, lora_request: %s.",
                request_id,
                params,
                lora_request,
            )
        self._write_io_log(
            "input",
            {
                "request_id": request_id,
                "prompt": prompt,
                "prompt_token_ids": prompt_token_ids,
                "prompt_embeds_shape": (
                    list(prompt_embeds.shape) if prompt_embeds is not None else None
                ),
                "params": params,
                "lora_request": lora_request,
            },
        )

    def log_outputs(
        self,
        request_id: str,
        outputs: str,
        output_token_ids: Sequence[int] | None,
        finish_reason: str | None = None,
        is_streaming: bool = False,
        delta: bool = False,
    ) -> None:
        outputs_for_log = outputs
        output_token_ids_for_log = output_token_ids
        max_log_len = self.max_log_len
        if max_log_len is not None:
            if outputs_for_log is not None:
                outputs_for_log = outputs_for_log[:max_log_len]

            if output_token_ids_for_log is not None:
                # Convert to list and apply truncation
                output_token_ids_for_log = list(output_token_ids_for_log)[:max_log_len]

        stream_info = ""
        if is_streaming:
            stream_info = " (streaming delta)" if delta else " (streaming complete)"

        if self.enable_log_outputs:
            logger.info(
                "Generated response %s%s: output: %r, "
                "output_token_ids: %s, finish_reason: %s",
                request_id,
                stream_info,
                outputs_for_log,
                output_token_ids_for_log,
                finish_reason,
            )

        if not delta:
            self._write_io_log(
                "output",
                {
                    "request_id": request_id,
                    "outputs": outputs,
                    "output_token_ids": output_token_ids,
                    "finish_reason": finish_reason,
                    "is_streaming": is_streaming,
                    "delta": delta,
                },
            )
