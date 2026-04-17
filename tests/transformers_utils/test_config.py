# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

import torch

from vllm.config import ModelConfig
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import (
    get_config,
    maybe_override_with_speculators,
    try_get_generation_config,
)
from vllm.transformers_utils.configs.rwkv7 import RWKV7Config


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118


def test_get_config_infers_rwkv7_native_pth_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "rwkv7-g1e-1.5b-20260309-ctx8192.pth"
    state_dict = {
        "emb.weight": torch.zeros(48, 128),
        "blocks.0.att.receptance.weight": torch.zeros(128, 128),
        "blocks.0.att.output.weight": torch.zeros(128, 128),
        "blocks.0.att.r_k": torch.zeros(2, 64),
        "blocks.0.att.w1": torch.zeros(128, 24),
        "blocks.0.att.a1": torch.zeros(128, 16),
        "blocks.0.att.g1": torch.zeros(128, 32),
        "blocks.0.ffn.key.weight": torch.zeros(512, 128),
        "blocks.1.att.output.weight": torch.zeros(128, 128),
        "blocks.1.att.v1": torch.zeros(128, 12),
        "blocks.1.ffn.key.weight": torch.zeros(512, 128),
        "ln_out.weight": torch.ones(128),
        "ln_out.bias": torch.zeros(128),
        "head.weight": torch.zeros(48, 128),
    }
    torch.save(state_dict, checkpoint_path)

    config = get_config(str(checkpoint_path), trust_remote_code=False)

    assert config.model_type == "rwkv7"
    assert config.architectures == ["RWKV7ForCausalLM"]
    assert config.vocab_size == 48
    assert config.hidden_size == 128
    assert config.num_hidden_layers == 2
    assert config.num_heads == 2
    assert config.head_dim == 64
    assert config.hidden_ratio == 4
    assert config.intermediate_size == 512
    assert config.decay_low_rank_dim == 24
    assert config.a_low_rank_dim == 16
    assert config.gate_low_rank_dim == 32
    assert config.v_low_rank_dim == 12
    assert config.value_dim == [128, 128]
    assert config.max_position_embeddings == 8192
    assert config.norm_bias is True


def test_model_config_accepts_rwkv7_native_pth_with_hf_tokenizer_dir(tmp_path):
    checkpoint_path = tmp_path / "rwkv7-g1e-1.5b-20260309-ctx8192.pth"
    torch.save(
        {
            "emb.weight": torch.zeros(48, 128),
            "blocks.0.att.receptance.weight": torch.zeros(128, 128),
            "blocks.0.att.output.weight": torch.zeros(128, 128),
            "blocks.0.att.r_k": torch.zeros(2, 64),
            "blocks.0.att.w1": torch.zeros(128, 24),
            "blocks.0.att.a1": torch.zeros(128, 16),
            "blocks.0.att.g1": torch.zeros(128, 32),
            "blocks.0.ffn.key.weight": torch.zeros(512, 128),
            "blocks.1.att.output.weight": torch.zeros(128, 128),
            "blocks.1.att.v1": torch.zeros(128, 12),
            "blocks.1.ffn.key.weight": torch.zeros(512, 128),
            "ln_out.weight": torch.ones(128),
            "ln_out.bias": torch.zeros(128),
            "head.weight": torch.zeros(48, 128),
        },
        checkpoint_path,
    )

    tokenizer_dir = tmp_path / "rwkv7-tokenizer"
    tokenizer_dir.mkdir()
    config = RWKV7Config(
        vocab_size=48,
        hidden_size=128,
        num_hidden_layers=2,
        num_heads=2,
        head_dim=64,
        hidden_ratio=4,
        intermediate_size=512,
        decay_low_rank_dim=24,
        a_low_rank_dim=16,
        gate_low_rank_dim=32,
        v_low_rank_dim=12,
        max_position_embeddings=8192,
        value_dim=[128, 128],
        architectures=["RWKV7ForCausalLM"],
    )
    (tokenizer_dir / "config.json").write_text(
        config.to_json_string(), encoding="utf-8"
    )

    model_config = ModelConfig(
        str(checkpoint_path),
        trust_remote_code=False,
        dtype="float32",
        runner="generate",
        tokenizer=str(tokenizer_dir),
    )

    assert model_config.model == str(checkpoint_path)
    assert model_config.hf_config.model_type == "rwkv7"
    assert model_config.hf_config.hidden_size == 128


def test_model_config_ignores_txt_tokenizer_as_hf_metadata_source(tmp_path):
    checkpoint_path = tmp_path / "rwkv7-g1e-1.5b-20260309-ctx8192.pth"
    torch.save(
        {
            "emb.weight": torch.zeros(48, 128),
            "blocks.0.att.receptance.weight": torch.zeros(128, 128),
            "blocks.0.att.output.weight": torch.zeros(128, 128),
            "blocks.0.att.r_k": torch.zeros(2, 64),
            "blocks.0.att.w1": torch.zeros(128, 24),
            "blocks.0.att.a1": torch.zeros(128, 16),
            "blocks.0.att.g1": torch.zeros(128, 32),
            "blocks.0.ffn.key.weight": torch.zeros(512, 128),
            "blocks.1.att.output.weight": torch.zeros(128, 128),
            "blocks.1.att.v1": torch.zeros(128, 12),
            "blocks.1.ffn.key.weight": torch.zeros(512, 128),
            "ln_out.weight": torch.ones(128),
            "ln_out.bias": torch.zeros(128),
            "head.weight": torch.zeros(48, 128),
        },
        checkpoint_path,
    )
    tokenizer_path = tmp_path / "rwkv_vocab_v20250609.txt"
    tokenizer_path.write_text(
        "1 'a' 1\n2 '\\n\\n' 2\n3 '<|endoftext|>' 13\n", encoding="utf-8"
    )

    model_config = ModelConfig(
        str(checkpoint_path),
        trust_remote_code=False,
        dtype="float32",
        runner="generate",
        tokenizer=str(tokenizer_path),
    )

    assert model_config.hf_config.model_type == "rwkv7"
    assert model_config.hf_config.hidden_size == 128
    assert model_config.encoder_config is None
    assert model_config.hf_image_processor_config == {}


def test_maybe_override_with_speculators_skips_native_pt_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "rwkv7-native.pth"
    torch.save({"emb.weight": torch.zeros(2, 2)}, checkpoint_path)

    model, tokenizer, speculative_config = maybe_override_with_speculators(
        model=str(checkpoint_path),
        tokenizer="dummy-tokenizer",
        trust_remote_code=False,
        vllm_speculative_config=None,
    )

    assert model == str(checkpoint_path)
    assert tokenizer == "dummy-tokenizer"
    assert speculative_config is None
