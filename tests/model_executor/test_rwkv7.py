# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.model_executor.models.rwkv7 import RWKV7Block, RWKV7ForCausalLM
from vllm.transformers_utils.configs.rwkv7 import RWKV7Config
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

try:
    import pytest
except ImportError:
    pytest = None


def _make_config() -> RWKV7Config:
    return RWKV7Config(
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


def _initialize_module_parameters(module: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(0)
    for name, parameter in module.named_parameters():
        if parameter.ndim == 0:
            parameter.data.zero_()
        elif name.endswith(".bias"):
            parameter.data.zero_()
        elif "g_norm.weight" in name:
            parameter.data.fill_(1.0)
        elif "k_a" in name:
            parameter.data.fill_(1.0)
        else:
            parameter.data.normal_(mean=0.0, std=0.02, generator=generator)


def _make_prefill_metadata(seq_len: int, *, device: torch.device) -> LinearAttentionMetadata:
    return LinearAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=seq_len,
        num_decodes=0,
        num_decode_tokens=0,
        query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        state_indices_tensor=torch.tensor([0], dtype=torch.long, device=device),
    )


def _make_decode_metadata(
    total_seq_len: int, *, device: torch.device
) -> LinearAttentionMetadata:
    return LinearAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([total_seq_len], dtype=torch.int32, device=device),
        state_indices_tensor=torch.tensor([0], dtype=torch.long, device=device),
    )


def _make_multi_decode_metadata(
    total_seq_lens: list[int], state_indices: list[int], *, device: torch.device
) -> LinearAttentionMetadata:
    num_decodes = len(total_seq_lens)
    return LinearAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        query_start_loc=torch.arange(
            0, num_decodes + 1, dtype=torch.int32, device=device
        ),
        seq_lens=torch.tensor(total_seq_lens, dtype=torch.int32, device=device),
        state_indices_tensor=torch.tensor(state_indices, dtype=torch.long, device=device),
    )


def _require_reference_checkpoint() -> tuple[Path, object]:
    if pytest is None:
        raise RuntimeError("pytest is required to run RWKV7 integration tests.")

    model_path = os.getenv("VLLM_RWKV7_TEST_MODEL_PATH")
    fla_path = os.getenv("VLLM_RWKV7_TEST_FLA_PATH")

    if not model_path:
        pytest.skip("Set VLLM_RWKV7_TEST_MODEL_PATH to run RWKV7 parity tests.")
    if not fla_path:
        pytest.skip("Set VLLM_RWKV7_TEST_FLA_PATH to run RWKV7 parity tests.")

    model_dir = Path(model_path)
    fla_dir = Path(fla_path)
    if not model_dir.exists():
        pytest.skip(f"RWKV7 model path does not exist: {model_dir}")
    if not fla_dir.exists():
        pytest.skip(f"FLA path does not exist: {fla_dir}")

    if str(fla_dir) not in sys.path:
        sys.path.insert(0, str(fla_dir))

    from fla.models.rwkv7 import RWKV7ForCausalLM as ReferenceRWKV7ForCausalLM

    return model_dir, ReferenceRWKV7ForCausalLM


def _make_vllm_config(model_path: Path) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(
            str(model_path),
            trust_remote_code=False,
            dtype="float32",
            runner="generate",
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        cache_config=CacheConfig(),
        device_config=DeviceConfig("cuda"),
    )


def _allocate_kv_cache(model: RWKV7ForCausalLM, *, device: torch.device) -> None:
    for layer in model.model.layers:
        state_shapes = layer.get_state_shape()
        state_dtypes = layer.get_state_dtype()
        layer.kv_cache = tuple(
            torch.zeros((1, *shape), dtype=dtype, device=device)
            for shape, dtype in zip(state_shapes, state_dtypes)
        )


def test_rwkv7_block_forward_without_metadata():
    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            block0 = RWKV7Block(config=config, layer_idx=0, prefix="model.layers.0")
            block1 = RWKV7Block(config=config, layer_idx=1, prefix="model.layers.1")
            _initialize_module_parameters(block0)
            _initialize_module_parameters(block1)

            hidden_states = torch.randn(5, config.hidden_size)
            hidden_states, v_first = block0(hidden_states, None, None)
            hidden_states, v_first = block1(hidden_states, v_first, None)

            assert hidden_states.shape == (5, config.hidden_size)
            assert v_first.shape == (5, config.hidden_size)
            assert torch.isfinite(hidden_states).all()
            assert torch.isfinite(v_first).all()
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_block_registers_static_forward_context():
    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            prefix = "model.layers.0"
            block = RWKV7Block(config=config, layer_idx=0, prefix=prefix)
            assert (
                vllm_config.compilation_config.static_forward_context[prefix]
                is block
            )
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_block_updates_cached_states():
    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            block = RWKV7Block(config=config, layer_idx=0, prefix="model.layers.0")
            _initialize_module_parameters(block)

            block.kv_cache = (
                torch.zeros(1, config.hidden_size),
                torch.zeros(1, config.num_heads, config.head_dim, config.head_dim),
                torch.zeros(1, config.hidden_size),
            )

            prefill_metadata = _make_prefill_metadata(3, device=torch.device("cpu"))
            hidden_states = torch.randn(3, config.hidden_size)
            output, v_first = block(hidden_states, None, prefill_metadata)

            assert output.shape == hidden_states.shape
            assert v_first.shape == hidden_states.shape
            assert torch.isfinite(output).all()
            assert block.kv_cache[0][0].abs().sum() > 0
            assert block.kv_cache[1][0].abs().sum() > 0
            assert block.kv_cache[2][0].abs().sum() > 0

            decode_metadata = _make_decode_metadata(4, device=torch.device("cpu"))
            decode_hidden = torch.randn(1, config.hidden_size)
            decode_output, decode_v_first = block(
                decode_hidden, v_first[:1].clone(), decode_metadata
            )

            assert decode_output.shape == decode_hidden.shape
            assert decode_v_first.shape == decode_hidden.shape
            assert torch.isfinite(decode_output).all()
            assert torch.isfinite(decode_v_first).all()
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_block_batches_decode_tokens_without_changing_results():
    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            block_batched = RWKV7Block(
                config=config, layer_idx=0, prefix="model.layers.0"
            )
            _initialize_module_parameters(block_batched)
            block_ref = RWKV7Block(config=config, layer_idx=0, prefix="model.layers.1")
            block_ref.load_state_dict(block_batched.state_dict())

            generator = torch.Generator().manual_seed(123)
            state_shapes = block_batched.get_state_shape()
            state_dtypes = block_batched.get_state_dtype()

            def make_cache() -> tuple[torch.Tensor, ...]:
                return tuple(
                    torch.randn(
                        (2, *shape),
                        generator=generator,
                        dtype=dtype,
                    )
                    for shape, dtype in zip(state_shapes, state_dtypes)
                )

            block_batched.kv_cache = make_cache()
            block_ref.kv_cache = tuple(cache.clone() for cache in block_batched.kv_cache)

            hidden_states = torch.randn(
                2, config.hidden_size, generator=generator, dtype=torch.float32
            )
            metadata = _make_multi_decode_metadata(
                [5, 7], [0, 1], device=torch.device("cpu")
            )

            output_batched, v_first_batched = block_batched(
                hidden_states, None, metadata
            )

            output_ref = torch.empty_like(hidden_states)
            v_first_ref = torch.empty_like(hidden_states)
            for idx, slot_id in enumerate([0, 1]):
                states = block_ref._get_kv_state(slot_id, use_initial_state=True)
                out, v_first_out, attn_shift, recurrent, ffn_shift = block_ref._run_sequence(
                    hidden_states[idx : idx + 1],
                    None,
                    *states,
                )
                output_ref[idx : idx + 1] = out
                v_first_ref[idx : idx + 1] = v_first_out
                block_ref._store_kv_state(slot_id, attn_shift, recurrent, ffn_shift)

            torch.testing.assert_close(output_batched, output_ref)
            torch.testing.assert_close(v_first_batched, v_first_ref)
            for batched_state, ref_state in zip(block_batched.kv_cache, block_ref.kv_cache):
                torch.testing.assert_close(batched_state, ref_state)
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_mamba_state_copy_function_types():
    copy_funcs = RWKV7ForCausalLM.get_mamba_state_copy_func()
    assert copy_funcs == (
        get_conv_copy_spec,
        get_temporal_copy_spec,
        get_conv_copy_spec,
    )


def test_rwkv7_block_uses_fp32_runtime_state_dtype():
    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            block = RWKV7Block(config=config, layer_idx=0, prefix="model.layers.0")
            assert block.get_state_dtype() == (
                torch.float32,
                torch.float32,
                torch.float32,
            )
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_reference_parity_full_forward():
    if pytest is None:
        raise RuntimeError("pytest is required to run RWKV7 integration tests.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RWKV7 reference parity tests.")

    model_path, reference_cls = _require_reference_checkpoint()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    inputs = tokenizer(
        "Hello RWKV7, this is a parity check.", return_tensors="pt"
    )["input_ids"].to("cuda")
    flat_input_ids = inputs[0]
    positions = torch.arange(flat_input_ids.numel(), device="cuda", dtype=torch.long)

    reference_model = reference_cls.from_pretrained(
        model_path, dtype=torch.float32
    ).eval().to("cuda")
    vllm_config = _make_vllm_config(model_path)

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="nccl",
        )
        ensure_model_parallel_initialized(1, 1, backend="nccl")
        try:
            vllm_model = RWKV7ForCausalLM(vllm_config=vllm_config)
            vllm_model.load_weights(reference_model.state_dict().items())
            vllm_model = vllm_model.eval().to("cuda", torch.float32)

            with torch.no_grad():
                reference_outputs = reference_model(
                    input_ids=inputs,
                    use_cache=False,
                )
                reference_hidden = reference_model.model(
                    input_ids=inputs,
                    use_cache=False,
                )[0][0]

            with torch.no_grad(), set_forward_context(None, vllm_config):
                hidden_states = vllm_model(
                    input_ids=flat_input_ids,
                    positions=positions,
                )
                logits = vllm_model.compute_logits(hidden_states)

            hidden_diff = (hidden_states - reference_hidden).abs()
            logits_diff = (logits - reference_outputs.logits[0]).abs()
            assert hidden_diff.max().item() < 5e-5
            assert logits_diff.max().item() < 5e-5
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_reference_parity_prefill_decode():
    if pytest is None:
        raise RuntimeError("pytest is required to run RWKV7 integration tests.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RWKV7 reference parity tests.")

    model_path, reference_cls = _require_reference_checkpoint()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt_ids = tokenizer(
        "The capital of France is", return_tensors="pt"
    )["input_ids"].to("cuda")

    reference_model = reference_cls.from_pretrained(
        model_path, dtype=torch.float32
    ).eval().to("cuda")
    vllm_config = _make_vllm_config(model_path)

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="nccl",
        )
        ensure_model_parallel_initialized(1, 1, backend="nccl")
        try:
            vllm_model = RWKV7ForCausalLM(vllm_config=vllm_config)
            vllm_model.load_weights(reference_model.state_dict().items())
            vllm_model = vllm_model.eval().to("cuda", torch.float32)
            _allocate_kv_cache(vllm_model, device=torch.device("cuda"))

            prompt_flat = prompt_ids[0]
            prompt_positions = torch.arange(
                prompt_flat.numel(), device="cuda", dtype=torch.long
            )
            prompt_metadata = {
                layer.prefix: _make_prefill_metadata(
                    prompt_ids.shape[1], device=torch.device("cuda")
                )
                for layer in vllm_model.model.layers
            }

            with torch.no_grad():
                reference_prompt_logits = reference_model(
                    input_ids=prompt_ids, use_cache=False
                ).logits[0]

            with torch.no_grad(), set_forward_context(prompt_metadata, vllm_config):
                hidden_states = vllm_model(
                    input_ids=prompt_flat,
                    positions=prompt_positions,
                )
                logits = vllm_model.compute_logits(hidden_states)

            assert (logits - reference_prompt_logits).abs().max().item() < 5e-5

            next_token = logits[-1].argmax().view(1)
            reference_first_token = reference_prompt_logits[-1].argmax().view(1)
            assert int(next_token.item()) == int(reference_first_token.item())

            generated_vllm = [int(next_token.item())]
            generated_ref = [int(reference_first_token.item())]
            current_ids = prompt_ids.clone()

            for _ in range(3):
                total_seq_len = current_ids.shape[1] + 1
                decode_metadata = {
                    layer.prefix: _make_decode_metadata(
                        total_seq_len, device=torch.device("cuda")
                    )
                    for layer in vllm_model.model.layers
                }
                position = torch.tensor(
                    [current_ids.shape[1]], device="cuda", dtype=torch.long
                )

                with torch.no_grad(), set_forward_context(
                    decode_metadata, vllm_config
                ):
                    hidden_states = vllm_model(
                        input_ids=next_token,
                        positions=position,
                    )
                    logits = vllm_model.compute_logits(hidden_states)

                full_ids = torch.cat([current_ids, next_token.view(1, 1)], dim=1)
                with torch.no_grad():
                    reference_last_logits = reference_model(
                        input_ids=full_ids,
                        use_cache=False,
                    ).logits[0, -1]

                assert (logits[-1] - reference_last_logits).abs().max().item() < 5e-5

                next_token = logits[-1].argmax().view(1)
                reference_next_token = reference_last_logits.argmax().view(1)
                assert int(next_token.item()) == int(reference_next_token.item())

                generated_vllm.append(int(next_token.item()))
                generated_ref.append(int(reference_next_token.item()))
                current_ids = full_ids

            assert generated_vllm == generated_ref
            assert tokenizer.decode(generated_vllm) == tokenizer.decode(generated_ref)
        finally:
            cleanup_dist_env_and_memory()
