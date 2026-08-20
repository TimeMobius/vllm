#include <torch/all.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

namespace {

constexpr int kRWKV7MaskedStoreThreads = 256;

template <typename scalar_t>
__global__ void rwkv7_masked_store_kernel(
    scalar_t* __restrict__ cache, const scalar_t* __restrict__ values,
    const int64_t* __restrict__ slot_ids, const int64_t row_numel) {
  const int64_t value_row = blockIdx.y;
  const int64_t slot_id = slot_ids[value_row];
  if (slot_id < 0) {
    return;
  }

  const int64_t offset =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (offset >= row_numel) {
    return;
  }

  cache[slot_id * row_numel + offset] = values[value_row * row_numel + offset];
}

}  // namespace

void rwkv7_masked_store(torch::Tensor& cache, const torch::Tensor& values,
                        const torch::Tensor& slot_ids) {
  TORCH_CHECK(cache.is_cuda(), "`cache` must be a CUDA tensor.");
  TORCH_CHECK(values.is_cuda(), "`values` must be a CUDA tensor.");
  TORCH_CHECK(slot_ids.is_cuda(), "`slot_ids` must be a CUDA tensor.");
  TORCH_CHECK(cache.scalar_type() == values.scalar_type(),
              "`cache` and `values` must have the same dtype.");
  TORCH_CHECK(slot_ids.scalar_type() == torch::kInt64,
              "`slot_ids` must have dtype int64.");
  TORCH_CHECK(cache.is_contiguous(), "`cache` must be contiguous.");
  TORCH_CHECK(values.is_contiguous(), "`values` must be contiguous.");
  TORCH_CHECK(slot_ids.is_contiguous(), "`slot_ids` must be contiguous.");
  TORCH_CHECK(cache.dim() >= 1 && values.dim() == cache.dim(),
              "`cache` and `values` must have the same rank >= 1.");
  TORCH_CHECK(values.size(0) == slot_ids.numel(),
              "`values.size(0)` must equal `slot_ids.numel()`.");
  TORCH_CHECK(cache.sizes().slice(1) == values.sizes().slice(1),
              "`cache` and `values` must have matching row shapes.");

  const auto batch_size = values.size(0);
  if (batch_size == 0) {
    return;
  }

  const auto row_numel = values.numel() / batch_size;
  const dim3 block(kRWKV7MaskedStoreThreads);
  const dim3 grid((row_numel + kRWKV7MaskedStoreThreads - 1) /
                      kRWKV7MaskedStoreThreads,
                  batch_size);
  const at::cuda::OptionalCUDAGuard device_guard(values.device());
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, values.scalar_type(), "rwkv7_masked_store", [&] {
        rwkv7_masked_store_kernel<scalar_t><<<grid, block, 0, stream>>>(
            cache.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(),
            slot_ids.data_ptr<int64_t>(), row_numel);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
