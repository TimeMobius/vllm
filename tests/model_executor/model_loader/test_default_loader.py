# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


def test_default_loader_accepts_single_pth_checkpoint_in_auto_mode(tmp_path):
    checkpoint_path = tmp_path / "rwkv7-g1e-1.5b-20260309-ctx8192.pth"
    torch.save({"dummy.weight": torch.zeros(1)}, checkpoint_path)

    loader = DefaultModelLoader(LoadConfig(load_format="auto"))
    hf_folder, checkpoint_files, use_safetensors = loader._prepare_weights(
        str(checkpoint_path),
        subfolder=None,
        revision=None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )

    assert hf_folder == str(tmp_path)
    assert checkpoint_files == [str(checkpoint_path)]
    assert use_safetensors is False
