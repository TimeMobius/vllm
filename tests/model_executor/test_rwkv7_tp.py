# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel regression coverage for RWKV7."""

import json
import os
from pathlib import Path

import pytest
import torch

import vllm.model_executor.models.rwkv7 as rwkv7_model
from tests.utils import multi_gpu_test
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CompilationConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.rwkv7 import RWKV7ForCausalLM
from vllm.transformers_utils.configs.rwkv7 import RWKV7Config
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

_TP_SIZE = 2


def _make_config(**overrides) -> RWKV7Config:
    config = dict(
        vocab_size=128,
        hidden_size=64,
        hidden_ratio=2,
        num_hidden_layers=2,
        head_dim=16,
        num_heads=4,
        decay_low_rank_dim=16,
        gate_low_rank_dim=16,
        a_low_rank_dim=16,
        v_low_rank_dim=16,
        norm_bias=True,
        value_dim=64,
    )
    config.update(overrides)
    return RWKV7Config(**config)


def _write_config_dir(tmp_path: Path) -> Path:
    model_path = tmp_path / "rwkv7-tp-model"
    model_path.mkdir()
    config_dict = _make_config().to_dict()
    config_dict["architectures"] = ["RWKV7ForCausalLM"]
    (model_path / "config.json").write_text(json.dumps(config_dict), encoding="utf-8")
    return model_path


def _make_vllm_config(model_path: str, tensor_parallel_size: int) -> VllmConfig:
    # This regression checks TP math and weight sharding, not torch.compile.
    # Eager mode also avoids compiling a separate graph in every spawned rank.
    return VllmConfig(
        model_config=ModelConfig(
            model_path,
            trust_remote_code=False,
            dtype="float32",
            runner="generate",
            enforce_eager=True,
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=1,
        ),
        cache_config=CacheConfig(),
        device_config=DeviceConfig("cuda"),
        compilation_config=CompilationConfig(mode=0),
    )


def _initialize_parameters(module: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(9821)
    for name, parameter in module.named_parameters():
        if parameter.ndim == 0 or name.endswith(".bias"):
            parameter.data.zero_()
        elif "g_norm.weight" in name or name.endswith(".k_a"):
            parameter.data.fill_(1.0)
        else:
            parameter.data.normal_(mean=0.0, std=0.02, generator=generator)


def _allocate_kv_cache(model: RWKV7ForCausalLM, device: torch.device) -> None:
    for layer in model.model.layers:
        layer.kv_cache = tuple(
            torch.zeros((1, *shape), dtype=dtype, device=device)
            for shape, dtype in zip(layer.get_state_shape(), layer.get_state_dtype())
        )


def _prefill_metadata(seq_len: int, device: torch.device) -> LinearAttentionMetadata:
    return LinearAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=seq_len,
        num_decodes=0,
        num_decode_tokens=0,
        query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        state_indices_tensor=torch.tensor([0], dtype=torch.long, device=device),
    )


def _decode_metadata(seq_len: int, device: torch.device) -> LinearAttentionMetadata:
    return LinearAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        state_indices_tensor=torch.tensor([0], dtype=torch.long, device=device),
    )


def _metadata_for_model(
    model: RWKV7ForCausalLM,
    metadata: LinearAttentionMetadata,
) -> dict[str, LinearAttentionMetadata]:
    return {layer.prefix: metadata for layer in model.model.layers}


def _assert_tp_weight_shards(
    model: RWKV7ForCausalLM,
    source_weights: dict[str, torch.Tensor],
    rank: int,
    world_size: int,
) -> None:
    sharded_dimensions = {
        "model.embed_tokens.weight": 0,
        "lm_head.weight": 0,
    }
    for layer_idx in range(2):
        prefix = f"model.layers.{layer_idx}"
        sharded_dimensions.update(
            {
                f"{prefix}.attn.r_proj.weight": 0,
                f"{prefix}.attn.k_proj.weight": 0,
                f"{prefix}.attn.v_proj.weight": 0,
                f"{prefix}.attn.o_proj.weight": 1,
                f"{prefix}.attn.w_lora.lora.2.weight": 0,
                f"{prefix}.attn.a_lora.lora.2.weight": 0,
                f"{prefix}.attn.g_lora.lora.2.weight": 0,
                f"{prefix}.ffn.key.weight": 0,
                f"{prefix}.ffn.value.weight": 1,
            }
        )
        if layer_idx != 0:
            sharded_dimensions[f"{prefix}.attn.v_lora.lora.2.weight"] = 0

    parameters = dict(model.named_parameters())
    for name, dim in sharded_dimensions.items():
        expected = source_weights[name].chunk(world_size, dim=dim)[rank]
        torch.testing.assert_close(parameters[name].detach().cpu(), expected)


def _tp_parity_worker(
    local_rank: int,
    world_size: int,
    port: int,
    model_path: str,
) -> None:
    # Keep this regression focused on sharding. Dedicated tests cover fused ops;
    # direct-linear is intentionally incompatible with TP.
    for name in (
        "RWKV7_USE_FUSED_MIX6",
        "RWKV7_USE_FUSED_KK_PRE",
        "RWKV7_USE_FUSED_LNX_RKVRES_XG",
        "RWKV7_USE_FUSED_CMIX",
        "RWKV7_USE_ALT_RECURRENT_KERNEL",
        "RWKV7_USE_DIRECT_LINEAR",
    ):
        os.environ[name] = "0"
    os.environ["VLLM_ALLREDUCE_USE_FLASHINFER"] = "0"
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )

    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )

    try:
        reference_config = _make_vllm_config(model_path, tensor_parallel_size=1)
        with set_current_vllm_config(reference_config):
            # Every rank builds the same TP=1 reference in a singleton group.
            initialize_model_parallel(1, 1, backend="nccl")
            reference_model = RWKV7ForCausalLM(vllm_config=reference_config).eval()
            _initialize_parameters(reference_model)
            source_weights = {
                name: parameter.detach().cpu().clone()
                for name, parameter in reference_model.named_parameters()
            }
            reference_model = reference_model.to(device)
            _allocate_kv_cache(reference_model, device)

            input_ids = torch.tensor([2, 7, 13, 29, 3], dtype=torch.long, device=device)
            positions = torch.arange(input_ids.numel(), dtype=torch.long, device=device)
            prefill_metadata = _metadata_for_model(
                reference_model,
                _prefill_metadata(input_ids.numel(), device),
            )
            with (
                torch.no_grad(),
                set_forward_context(prefill_metadata, reference_config),
            ):
                reference_hidden = reference_model(input_ids, positions)
                reference_logits = reference_model.compute_logits(reference_hidden)

            reference_next_token = reference_logits[-1].argmax().view(1)
            decode_metadata = _metadata_for_model(
                reference_model,
                _decode_metadata(input_ids.numel() + 1, device),
            )
            with (
                torch.no_grad(),
                set_forward_context(decode_metadata, reference_config),
            ):
                reference_decode_hidden = reference_model(
                    reference_next_token,
                    torch.tensor([input_ids.numel()], dtype=torch.long, device=device),
                )
                reference_decode_logits = reference_model.compute_logits(
                    reference_decode_hidden
                )

            reference_hidden = reference_hidden.cpu()
            reference_logits = reference_logits.cpu()
            reference_decode_hidden = reference_decode_hidden.cpu()
            reference_decode_logits = reference_decode_logits.cpu()

        del reference_model
        destroy_model_parallel()
        torch.accelerator.empty_cache()

        tp_config = _make_vllm_config(model_path, tensor_parallel_size=world_size)
        with set_current_vllm_config(tp_config):
            initialize_model_parallel(world_size, 1, backend="nccl")
            tp_model = RWKV7ForCausalLM(vllm_config=tp_config).eval().to(device)
            loaded_weights = tp_model.load_weights(source_weights.items())
            assert loaded_weights == set(source_weights)
            _assert_tp_weight_shards(
                tp_model,
                source_weights,
                rank=local_rank,
                world_size=world_size,
            )
            _allocate_kv_cache(tp_model, device)

            prefill_metadata = _metadata_for_model(
                tp_model,
                _prefill_metadata(input_ids.numel(), device),
            )
            with torch.no_grad(), set_forward_context(prefill_metadata, tp_config):
                tp_hidden = tp_model(input_ids, positions)
                tp_logits = tp_model.compute_logits(tp_hidden)

            torch.testing.assert_close(tp_hidden.cpu(), reference_hidden)
            torch.testing.assert_close(tp_logits.cpu(), reference_logits)

            next_token = tp_logits[-1].argmax().view(1)
            assert int(next_token.item()) == int(reference_next_token.item())
            decode_metadata = _metadata_for_model(
                tp_model,
                _decode_metadata(input_ids.numel() + 1, device),
            )
            with torch.no_grad(), set_forward_context(decode_metadata, tp_config):
                tp_decode_hidden = tp_model(
                    next_token,
                    torch.tensor([input_ids.numel()], dtype=torch.long, device=device),
                )
                tp_decode_logits = tp_model.compute_logits(tp_decode_hidden)

            torch.testing.assert_close(tp_decode_hidden.cpu(), reference_decode_hidden)
            torch.testing.assert_close(tp_decode_logits.cpu(), reference_decode_logits)
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (
            _make_config(hidden_size=96, head_dim=32, num_heads=3, value_dim=96),
            "num_heads",
        ),
        (_make_config(intermediate_size=129), "intermediate_size"),
    ],
)
def test_rwkv7_tp_dimension_validation(monkeypatch, config: RWKV7Config, error: str):
    monkeypatch.setattr(rwkv7_model, "get_tp_world_size", lambda: _TP_SIZE)

    with pytest.raises(ValueError, match=error):
        rwkv7_model._validate_rwkv7_tensor_parallel_config(config)


@multi_gpu_test(num_gpus=_TP_SIZE)
def test_rwkv7_tp2_matches_tp1_prefill_decode_and_weight_shards(tmp_path: Path):
    model_path = _write_config_dir(tmp_path)
    torch.multiprocessing.spawn(
        _tp_parity_worker,
        args=(_TP_SIZE, get_open_port(), str(model_path)),
        nprocs=_TP_SIZE,
        join=True,
    )
