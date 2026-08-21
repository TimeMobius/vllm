#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

namespace {
constexpr int kMaxWarps = 8;

__device__ __forceinline__ float add_rn(float a, float b) { return __fadd_rn(a, b); }

// Variant 0 is intentionally the closest source-level reconstruction of the
// historical strict prototype: one output row per CTA, scalar BF16 loads,
// explicitly rounded FP32 FMA, a warp tree, then a serial warp-partial tree.
template <int THREADS, bool VOLATILE_ACC, bool WARP_TREE>
__global__ void bf16_gemv_row_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int64_t out_features,
    int64_t in_features) {
  const int64_t row = blockIdx.x;
  if (row >= out_features) return;
  const int tid = threadIdx.x;
  const __nv_bfloat16* row_weight = weight + row * in_features;
  float acc = 0.0f;
#pragma unroll 1
  for (int64_t col = tid; col < in_features; col += THREADS) {
    const float x = __bfloat162float(input[col]);
    const float w = __bfloat162float(row_weight[col]);
    if constexpr (VOLATILE_ACC) {
      volatile float next = __fmaf_rn(x, w, acc);
      acc = next;
    } else {
      acc = __fmaf_rn(x, w, acc);
    }
  }
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if constexpr (WARP_TREE) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc = add_rn(acc, __shfl_down_sync(0xffffffff, acc, offset));
    }
  } else {
    // Same per-warp order as a scalar lane-ordered fold.
    for (int offset = 1; offset < 32; offset <<= 1) {
      const float other = __shfl_up_sync(0xffffffff, acc, offset);
      if (lane >= offset) acc = add_rn(acc, other);
    }
  }
  __shared__ float partial[kMaxWarps];
  if (lane == 0) partial[warp] = acc;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < THREADS / 32; ++w) total = add_rn(total, partial[w]);
    output[row] = __float2bfloat16_rn(total);
  }
}

template <int THREADS, bool VOLATILE_ACC>
__global__ void bf16_gemv_row_four_acc_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int64_t out_features,
    int64_t in_features) {
  const int64_t row = blockIdx.x;
  if (row >= out_features) return;
  const int tid = threadIdx.x;
  const __nv_bfloat16* row_weight = weight + row * in_features;
  float accum[4] = {};
#pragma unroll 1
  for (int64_t col = tid; col < in_features; col += THREADS * 4) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t c = col + i * THREADS;
      if (c < in_features) {
        const float x = __bfloat162float(input[c]);
        const float w = __bfloat162float(row_weight[c]);
        if constexpr (VOLATILE_ACC) {
          volatile float next = __fmaf_rn(x, w, accum[i]);
          accum[i] = next;
        } else {
          accum[i] = __fmaf_rn(x, w, accum[i]);
        }
      }
    }
  }
  float acc = add_rn(add_rn(accum[0], accum[1]), add_rn(accum[2], accum[3]));
  const int lane = tid & 31;
  const int warp = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc = add_rn(acc, __shfl_down_sync(0xffffffff, acc, offset));
  }
  __shared__ float partial[kMaxWarps];
  if (lane == 0) partial[warp] = acc;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < THREADS / 32; ++w) total = add_rn(total, partial[w]);
    output[row] = __float2bfloat16_rn(total);
  }
}


template <int THREADS, int ACCS, bool FINAL_WARP_TREE>
__global__ void bf16_gemv_row_multiacc_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int64_t out_features,
    int64_t in_features) {
  const int64_t row = blockIdx.x;
  if (row >= out_features) return;
  const int tid = threadIdx.x;
  const __nv_bfloat16* row_weight = weight + row * in_features;
  float accum[ACCS] = {};
#pragma unroll 1
  for (int64_t base = tid; base < in_features; base += THREADS * ACCS) {
#pragma unroll
    for (int i = 0; i < ACCS; ++i) {
      const int64_t col = base + static_cast<int64_t>(i) * THREADS;
      if (col < in_features) {
        accum[i] = __fmaf_rn(__bfloat162float(input[col]),
                              __bfloat162float(row_weight[col]), accum[i]);
      }
    }
  }
  // Fixed pairwise tree for 2/4/8 accumulators.
  for (int step = ACCS / 2; step > 0; step >>= 1) {
#pragma unroll
    for (int i = 0; i < step; ++i) accum[i] = add_rn(accum[i], accum[i + step]);
  }
  float acc = accum[0];
  const int lane = tid & 31;
  const int warp = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc = add_rn(acc, __shfl_down_sync(0xffffffff, acc, offset));
  }
  __shared__ float partial[16];
  if (lane == 0) partial[warp] = acc;
  __syncthreads();
  if constexpr (FINAL_WARP_TREE) {
    if (warp == 0) {
      float x = lane < THREADS / 32 ? partial[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        x = add_rn(x, __shfl_down_sync(0xffffffff, x, offset));
      }
      if (lane == 0) output[row] = __float2bfloat16_rn(x);
    }
  } else if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < THREADS / 32; ++w) total = add_rn(total, partial[w]);
    output[row] = __float2bfloat16_rn(total);
  }
}

void bf16_gemv_variant_out(
    at::Tensor& output, const at::Tensor& input, const at::Tensor& weight,
    int64_t variant) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && output.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16, "BF16 tensors required");
  TORCH_CHECK(input.dim() == 2 && input.size(0) == 1 && weight.dim() == 2 && output.dim() == 2, "expected [1,K], [N,K], [1,N]");
  TORCH_CHECK(input.size(1) == weight.size(1) && output.size(0) == 1 && output.size(1) == weight.size(0), "shape mismatch");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && output.is_contiguous(), "contiguous required");
  c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* x = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());
  const auto* w = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());
  auto* o = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
  const auto n = weight.size(0), k = weight.size(1);
  switch (variant) {
    case 0: bf16_gemv_row_kernel<128, false, true><<<n, 128, 0, stream>>>(x,w,o,n,k); break;
    case 1: bf16_gemv_row_kernel<128, true, true><<<n, 128, 0, stream>>>(x,w,o,n,k); break;
    case 2: bf16_gemv_row_kernel<256, false, true><<<n, 256, 0, stream>>>(x,w,o,n,k); break;
    case 3: bf16_gemv_row_kernel<256, true, true><<<n, 256, 0, stream>>>(x,w,o,n,k); break;
    case 4: bf16_gemv_row_four_acc_kernel<128, false><<<n, 128, 0, stream>>>(x,w,o,n,k); break;
    case 5: bf16_gemv_row_four_acc_kernel<128, true><<<n, 128, 0, stream>>>(x,w,o,n,k); break;
    case 6: bf16_gemv_row_four_acc_kernel<256, false><<<n, 256, 0, stream>>>(x,w,o,n,k); break;
    case 7: bf16_gemv_row_four_acc_kernel<256, true><<<n, 256, 0, stream>>>(x,w,o,n,k); break;
    case 8: bf16_gemv_row_multiacc_kernel<64, 2, false><<<n, 64, 0, stream>>>(x,w,o,n,k); break;
    case 9: bf16_gemv_row_multiacc_kernel<64, 4, false><<<n, 64, 0, stream>>>(x,w,o,n,k); break;
    case 10: bf16_gemv_row_multiacc_kernel<128, 2, false><<<n, 128, 0, stream>>>(x,w,o,n,k); break;
    case 11: bf16_gemv_row_multiacc_kernel<128, 8, false><<<n, 128, 0, stream>>>(x,w,o,n,k); break;
    case 12: bf16_gemv_row_multiacc_kernel<256, 2, false><<<n, 256, 0, stream>>>(x,w,o,n,k); break;
    case 13: bf16_gemv_row_multiacc_kernel<256, 4, false><<<n, 256, 0, stream>>>(x,w,o,n,k); break;
    case 14: bf16_gemv_row_multiacc_kernel<128, 4, true><<<n, 128, 0, stream>>>(x,w,o,n,k); break;
    case 15: bf16_gemv_row_multiacc_kernel<256, 4, true><<<n, 256, 0, stream>>>(x,w,o,n,k); break;
    default: TORCH_CHECK(false, "unknown variant");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace
TORCH_LIBRARY(rwkv7gemvprobe, m) {
  m.def("bf16_gemv_variant_out(Tensor(a!) output, Tensor input, Tensor weight, int variant) -> ()");
}
TORCH_LIBRARY_IMPL(rwkv7gemvprobe, CUDA, m) {
  m.impl("bf16_gemv_variant_out", TORCH_FN(bf16_gemv_variant_out));
}
