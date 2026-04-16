# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

import pytest

from vllm.compilation.cuda_graph import (
    _EMPTY_CUDA_GRAPH_WARNING_PREFIX,
    _handle_cuda_graph_capture_warnings,
)
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor


def _capture_warning(message: str) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        warnings.warn(message, UserWarning, stacklevel=1)
    return captured_warnings


def test_handle_cuda_graph_capture_warnings_suppresses_empty_graph_warning():
    captured_warnings = _capture_warning(
        _EMPTY_CUDA_GRAPH_WARNING_PREFIX
        + " This usually means capture happened on the wrong stream."
    )

    with warnings.catch_warnings(record=True) as reemitted_warnings:
        warnings.simplefilter("always")
        _handle_cuda_graph_capture_warnings(
            captured_warnings,
            runtime_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(num_tokens=128),
        )

    assert reemitted_warnings == []


def test_handle_cuda_graph_capture_warnings_reemits_unexpected_warning():
    captured_warnings = _capture_warning("some other cuda graph warning")

    with pytest.warns(UserWarning, match="some other cuda graph warning"):
        _handle_cuda_graph_capture_warnings(
            captured_warnings,
            runtime_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(num_tokens=128),
        )


def test_handle_cuda_graph_capture_warnings_does_not_suppress_full_mode():
    captured_warnings = _capture_warning(
        _EMPTY_CUDA_GRAPH_WARNING_PREFIX
        + " This should still surface outside piecewise mode."
    )

    with pytest.warns(UserWarning, match="The CUDA Graph is empty."):
        _handle_cuda_graph_capture_warnings(
            captured_warnings,
            runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=BatchDescriptor(num_tokens=128),
        )
