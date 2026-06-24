<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, Arm CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## RWKV7 (branch `codex/rwkv7-adapter-align`)

This branch ships an experimental RWKV7-Goose adapter together with optional fused
CUDA / Triton kernels. The pre-built wheels do **not** include the RWKV7 custom
ops (`_C.relu2`, `_C.rwkv7_alt_recurrent`) for every GPU architecture, so on a
fresh machine you should build vLLM from source for your local SM.

### 1. Build prerequisites (example: RTX 4090 / 4090 D, SM 8.9)

Pin the toolchain so that `nvcc`, host `g++`, sysroot and PyTorch are all
consistent. The example below uses a clean conda environment; adjust
`TORCH_CUDA_ARCH_LIST` for your card (`8.0` for A100, `8.9` for 4090/4090 D,
`9.0` for H100/H800, `10.0`/`12.0` for Blackwell).

```bash
# Fresh CPython env (skip if you already have one that is *not* GraalPy)
conda create -n vllm-cu124 python=3.12 -c conda-forge -y
conda activate vllm-cu124

# CUDA 12.4 toolkit (matches the cu124 PyTorch wheel)
conda install -y -c nvidia/label/cuda-12.4.1 \
  cuda-nvcc cuda-cudart cuda-cudart-dev cuda-cccl cuda-nvrtc cuda-nvrtc-dev \
  cuda-driver-dev libcusparse-dev libcublas-dev libcusolver-dev \
  libcurand-dev libcufft-dev

# Host compiler (gcc 13 + matching sysroot; CUDA 12.4 does not support gcc >= 14)
conda install -y -c conda-forge \
  "gcc_linux-64=13.*" "gxx_linux-64=13.*" binutils_linux-64 \
  "sysroot_linux-64=2.34"

# PyTorch (cu124) and build helpers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install setuptools_scm numpy ninja packaging cmake

# Conda lays cuda libs under $CONDA_PREFIX/lib; CMake looks for lib64
[ -e $CONDA_PREFIX/lib64 ] || ln -s $CONDA_PREFIX/lib $CONDA_PREFIX/lib64

# Wire up the toolchain for the build
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export CUDACXX=$(which nvcc)
export CUDAToolkit_ROOT=$CONDA_PREFIX
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
# Prevent /usr/lib64/ccache/c++ from re-injecting the system gcc
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '/usr/lib64/ccache' | paste -sd:)
```

Sanity-check before building:

```bash
python -c "import sys; print(sys.implementation.name)"   # must be 'cpython'
nvcc --version                                            # release 12.4
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
```

### 2. Build vLLM from source

```bash
cd /path/to/vllm                # this repo
rm -rf build/ dist/ *.egg-info .deps CMakeCache.txt CMakeFiles
find vllm -name "*.so" -delete

export TORCH_CUDA_ARCH_LIST="8.9"   # change for your GPU
export MAX_JOBS=8                   # raise if you have headroom; nvcc takes 4-8 GB / job

pip install -e . --no-build-isolation -v 2>&1 | tee /tmp/vllm_build.log
```

Verify the freshly built `_C` ops:

```bash
python - <<'PY'
import torch, vllm._C
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "cc", torch.cuda.get_device_capability())
x = torch.randn(1024, device="cuda"); out = torch.empty_like(x)
torch.ops._C.relu2(out, x); print("_C.relu2 ok")
H = 4
r = torch.randn(1, 1, H, 64, device="cuda", dtype=torch.float32).contiguous()
o, s = torch.ops._C.rwkv7_alt_recurrent(r, r, r, r, r, r, None)
print("_C.rwkv7_alt_recurrent ok", tuple(o.shape), tuple(s.shape))
PY
```

### 3. Serve an RWKV7 checkpoint

The command below enables every RWKV7 fused kernel that was confirmed
net-positive on the 0.4B isolated benchmark. All flags are fail-safe — if a
kernel is unavailable for your shape/dtype the model falls back to the Triton
reference path, so correctness is preserved.

```bash
CUDA_VISIBLE_DEVICES=0 \
RWKV7_USE_FUSED_MIX6=1 \
RWKV7_USE_FUSED_KK_PRE=1 \
RWKV7_USE_FUSED_LNX_RKVRES_XG=1 \
RWKV7_USE_ALT_RECURRENT_KERNEL=1 \
RWKV7_USE_FUSED_CMIX=1 \
RWKV7_USE_DIRECT_LINEAR=1 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve /path/to/rwkv7-checkpoint.pth \
  --served-model-name rwkv7 \
  --tokenizer /path/to/rwkv_vocab_v20260603.txt \
  --tokenizer-mode rwkv \
  --chat-template /path/to/chat_template.jinja \
  --reasoning-parser rwkv \
  --enable-auto-tool-choice --tool-call-parser rwkv \
  --load-format pt \
  --dtype bfloat16 \
  --host 0.0.0.0 --port 8030 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.92 \
  --enforce-eager \
  --enable-prefix-caching --mamba-cache-mode align
```

Notes:

- `--enforce-eager` is currently faster than piecewise CUDA graphs for RWKV7
  (decode TPS ≈ 1.4×). Drop it once piecewise wins on your workload.
- `RWKV7_USE_DIRECT_LINEAR=1` requires `tp_size == 1` and unquantized linears;
  long-prompt prefill (>1024 tokens) regresses slightly, so disable it for
  long-context-heavy workloads.
- `--mamba-cache-mode align` is a balanced choice for prefix caching; switch to
  `all` if your KV cache budget is generous and you need the highest hit rate.
- For HF-format RWKV7 checkpoints, drop `--load-format pt` and `--tokenizer*`
  flags; the auto loader will pick them up.

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
