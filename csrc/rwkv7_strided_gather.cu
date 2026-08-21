#include <torch/all.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace {

constexpr int kRWKV7StridedGatherThreads = 256;
constexpr int64_t kMaxGridYOrZ = 65535;
constexpr int64_t kMaxGridX = 65535;

template <typename scalar_t>
__global__ void rwkv7_strided_gather_kernel(
    scalar_t* __restrict__ output, const scalar_t* __restrict__ cache,
    const int64_t* __restrict__ slot_ids, const int64_t num_slots,
    const int64_t cache_rows, const int64_t cache_row_stride,
    const int64_t row_numel) {
  const int64_t output_row = static_cast<int64_t>(blockIdx.y) +
                             static_cast<int64_t>(blockIdx.z) * gridDim.y;
  if (output_row >= num_slots) {
    return;
  }

  const int64_t slot_id = slot_ids[output_row];
  // Matching index_select's bounds behavior without a host synchronization is
  // not possible for a CUDA-resident index. Fail the launch rather than issue
  // an out-of-bounds global-memory access for invalid input.
  assert(slot_id >= 0 && slot_id < cache_rows);

  const int64_t output_offset = output_row * row_numel;
  const int64_t cache_offset = slot_id * cache_row_stride;
  for (int64_t inner_offset =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       inner_offset < row_numel;
       inner_offset += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    // Direct scalar assignment is a bitwise copy for float, half, and
    // bfloat16. No arithmetic or dtype conversion is performed.
    output[output_offset + inner_offset] = cache[cache_offset + inner_offset];
  }
}

bool has_contiguous_inner_rows(const torch::Tensor& cache) {
  int64_t expected_stride = 1;
  for (int64_t dim = cache.dim() - 1; dim >= 1; --dim) {
    // A size-one dimension does not change addressing within a row, so its
    // stride may be arbitrary while the inner row is still contiguous.
    if (cache.size(dim) > 1 && cache.stride(dim) != expected_stride) {
      return false;
    }
    expected_stride *= cache.size(dim);
  }
  return true;
}

}  // namespace

torch::Tensor rwkv7_strided_gather(const torch::Tensor& cache,
                                   const torch::Tensor& slot_ids) {
  TORCH_CHECK(cache.is_cuda(), "`cache` must be a CUDA tensor.");
  TORCH_CHECK(slot_ids.is_cuda(), "`slot_ids` must be a CUDA tensor.");
  TORCH_CHECK(cache.device() == slot_ids.device(),
              "`cache` and `slot_ids` must be on the same CUDA device.");
  TORCH_CHECK(cache.dim() >= 1, "`cache` must have rank >= 1.");
  TORCH_CHECK(slot_ids.dim() == 1, "`slot_ids` must be 1D, got rank ",
              slot_ids.dim(), ".");
  TORCH_CHECK(slot_ids.scalar_type() == torch::kInt64,
              "`slot_ids` must have dtype int64.");
  TORCH_CHECK(slot_ids.is_contiguous(), "`slot_ids` must be contiguous.");
  TORCH_CHECK(cache.scalar_type() == torch::kFloat ||
                  cache.scalar_type() == torch::kHalf ||
                  cache.scalar_type() == torch::kBFloat16,
              "`cache` must have dtype float32, float16, or bfloat16; got ",
              cache.scalar_type(), ".");
  TORCH_CHECK(has_contiguous_inner_rows(cache),
              "`cache` must have contiguous dimensions 1..N-1.");

  const int64_t num_slots = slot_ids.numel();
  const int64_t cache_rows = cache.size(0);
  const int64_t row_numel = cache.numel() / std::max<int64_t>(cache_rows, 1);
  const int64_t cache_row_stride = cache.stride(0);
  TORCH_CHECK(row_numel == 0 || cache_row_stride >= row_numel,
              "`cache.stride(0)` must be at least the contiguous inner row "
              "size (", row_numel, "), got ", cache_row_stride, ".");

  const at::cuda::OptionalCUDAGuard device_guard(cache.device());
  std::vector<int64_t> output_sizes = cache.sizes().vec();
  output_sizes[0] = num_slots;
  auto output = torch::empty(
      output_sizes,
      cache.options().memory_format(torch::MemoryFormat::Contiguous));

  if (num_slots == 0 || row_numel == 0) {
    return output;
  }
  TORCH_CHECK(cache_rows > 0,
              "`cache.size(0)` must be nonzero when `slot_ids` is nonempty "
              "and each row has elements.");

  const int64_t grid_y = std::min<int64_t>(num_slots, kMaxGridYOrZ);
  const int64_t grid_z = (num_slots + grid_y - 1) / grid_y;
  TORCH_CHECK(grid_z <= kMaxGridYOrZ,
              "`slot_ids` has too many elements for the CUDA launch: ",
              num_slots, ".");

  const int64_t grid_x = std::min<int64_t>(
      (row_numel + kRWKV7StridedGatherThreads - 1) /
          kRWKV7StridedGatherThreads,
      kMaxGridX);
  const dim3 block(kRWKV7StridedGatherThreads);
  const dim3 grid(static_cast<uint32_t>(grid_x),
                  static_cast<uint32_t>(grid_y),
                  static_cast<uint32_t>(grid_z));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, cache.scalar_type(), "rwkv7_strided_gather",
      [&] {
        rwkv7_strided_gather_kernel<scalar_t><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), cache.data_ptr<scalar_t>(),
            slot_ids.data_ptr<int64_t>(), num_slots, cache_rows,
            cache_row_stride, row_numel);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
