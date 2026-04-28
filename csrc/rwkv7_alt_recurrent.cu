#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <optional>
#include <tuple>

namespace {

constexpr int64_t kRWKV7AltHeadDim = 64;

__global__ void rwkv7_alt_recurrent_kernel(
    const float* __restrict__ r, const float* __restrict__ w,
    const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ kk, const float* __restrict__ a,
    const float* __restrict__ initial_state, float* __restrict__ out,
    float* __restrict__ final_state, const int64_t batch_size,
    const int64_t seq_len, const int64_t num_heads) {
  const int64_t batch_idx = blockIdx.y;
  const int64_t head_idx = blockIdx.x;
  const int64_t value_idx = threadIdx.x;

  if (batch_idx >= batch_size || head_idx >= num_heads ||
      value_idx >= kRWKV7AltHeadDim) {
    return;
  }

  float state[kRWKV7AltHeadDim];
#pragma unroll
  for (int64_t key_idx = 0; key_idx < kRWKV7AltHeadDim; ++key_idx) {
    if (initial_state == nullptr) {
      state[key_idx] = 0.0f;
    } else {
      const int64_t offset =
          (((batch_idx * num_heads + head_idx) * kRWKV7AltHeadDim + key_idx) *
               kRWKV7AltHeadDim +
           value_idx);
      state[key_idx] = initial_state[offset];
    }
  }

  __shared__ float shared_r[kRWKV7AltHeadDim];
  __shared__ float shared_exp_w[kRWKV7AltHeadDim];
  __shared__ float shared_k[kRWKV7AltHeadDim];
  __shared__ float shared_neg_kk[kRWKV7AltHeadDim];
  __shared__ float shared_kk_a[kRWKV7AltHeadDim];

  for (int64_t token_idx = 0; token_idx < seq_len; ++token_idx) {
    const int64_t key_base =
        (((batch_idx * seq_len + token_idx) * num_heads + head_idx) *
         kRWKV7AltHeadDim);
    const int64_t value_base =
        (((batch_idx * seq_len + token_idx) * num_heads + head_idx) *
         kRWKV7AltHeadDim);

    __syncthreads();
    shared_r[value_idx] = r[key_base + value_idx];
    shared_exp_w[value_idx] = __expf(w[key_base + value_idx]);
    shared_k[value_idx] = k[key_base + value_idx];
    const float kk_val = kk[key_base + value_idx];
    shared_neg_kk[value_idx] = -kk_val;
    shared_kk_a[value_idx] = kk_val * a[key_base + value_idx];
    __syncthreads();

    float sa = 0.0f;
#pragma unroll
    for (int64_t key_idx = 0; key_idx < kRWKV7AltHeadDim; ++key_idx) {
      sa += state[key_idx] * shared_neg_kk[key_idx];
    }

    const float value_component = v[value_base + value_idx];
    float out_val = 0.0f;
#pragma unroll
    for (int64_t key_idx = 0; key_idx < kRWKV7AltHeadDim; ++key_idx) {
      const float updated_state = state[key_idx] * shared_exp_w[key_idx] +
                                  shared_kk_a[key_idx] * sa +
                                  shared_k[key_idx] * value_component;
      state[key_idx] = updated_state;
      out_val += updated_state * shared_r[key_idx];
    }

    out[value_base + value_idx] = out_val;
  }

#pragma unroll
  for (int64_t key_idx = 0; key_idx < kRWKV7AltHeadDim; ++key_idx) {
    const int64_t offset =
        (((batch_idx * num_heads + head_idx) * kRWKV7AltHeadDim + key_idx) *
             kRWKV7AltHeadDim +
         value_idx);
    final_state[offset] = state[key_idx];
  }
}

void check_rwkv7_alt_recurrent_tensor(const torch::Tensor& tensor,
                                      const char* name,
                                      const int64_t expected_batch_size,
                                      const int64_t expected_seq_len,
                                      const int64_t expected_num_heads,
                                      const int64_t expected_last_dim) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name,
              " must have dtype float32.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 4, name, " must be 4D, got ", tensor.dim(), ".");
  TORCH_CHECK(tensor.size(0) == expected_batch_size &&
                  tensor.size(1) == expected_seq_len &&
                  tensor.size(2) == expected_num_heads &&
                  tensor.size(3) == expected_last_dim,
              name, " must have shape [", expected_batch_size, ", ",
              expected_seq_len, ", ", expected_num_heads, ", ",
              expected_last_dim, "], got ", tensor.sizes(), ".");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> rwkv7_alt_recurrent(
    const torch::Tensor& r, const torch::Tensor& w, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& kk, const torch::Tensor& a,
    const std::optional<torch::Tensor>& initial_state) {
  TORCH_CHECK(r.is_cuda(), "`r` must be a CUDA tensor.");
  TORCH_CHECK(r.scalar_type() == torch::kFloat32,
              "`r` must have dtype float32.");
  TORCH_CHECK(r.is_contiguous(), "`r` must be contiguous.");
  TORCH_CHECK(r.dim() == 4, "`r` must be 4D, got ", r.dim(), ".");

  const int64_t batch_size = r.size(0);
  const int64_t seq_len = r.size(1);
  const int64_t num_heads = r.size(2);
  const int64_t head_dim = r.size(3);

  TORCH_CHECK(head_dim == kRWKV7AltHeadDim, "`r` must use head_dim=64, got ",
              head_dim, ".");

  check_rwkv7_alt_recurrent_tensor(w, "`w`", batch_size, seq_len, num_heads,
                                   kRWKV7AltHeadDim);
  check_rwkv7_alt_recurrent_tensor(k, "`k`", batch_size, seq_len, num_heads,
                                   kRWKV7AltHeadDim);
  check_rwkv7_alt_recurrent_tensor(kk, "`kk`", batch_size, seq_len, num_heads,
                                   kRWKV7AltHeadDim);
  check_rwkv7_alt_recurrent_tensor(a, "`a`", batch_size, seq_len, num_heads,
                                   kRWKV7AltHeadDim);
  check_rwkv7_alt_recurrent_tensor(v, "`v`", batch_size, seq_len, num_heads,
                                   kRWKV7AltHeadDim);

  if (initial_state.has_value()) {
    const torch::Tensor& h0 = *initial_state;
    TORCH_CHECK(h0.is_cuda(), "`initial_state` must be a CUDA tensor.");
    TORCH_CHECK(h0.scalar_type() == torch::kFloat32,
                "`initial_state` must have dtype float32.");
    TORCH_CHECK(h0.is_contiguous(), "`initial_state` must be contiguous.");
    TORCH_CHECK(h0.dim() == 4, "`initial_state` must be 4D, got ", h0.dim(),
                ".");
    TORCH_CHECK(h0.size(0) == batch_size && h0.size(1) == num_heads &&
                    h0.size(2) == kRWKV7AltHeadDim &&
                    h0.size(3) == kRWKV7AltHeadDim,
                "`initial_state` must have shape [", batch_size, ", ",
                num_heads, ", 64, 64], got ", h0.sizes(), ".");
  }

  c10::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(r.get_device());

  auto out = torch::empty_like(v, v.options().dtype(torch::kFloat32));
  auto final_state =
      torch::empty({batch_size, num_heads, kRWKV7AltHeadDim, kRWKV7AltHeadDim},
                   r.options().dtype(torch::kFloat32));

  const float* initial_state_ptr =
      initial_state.has_value() ? initial_state->data_ptr<float>() : nullptr;

  const dim3 grid(num_heads, batch_size);
  const dim3 block(kRWKV7AltHeadDim);
  rwkv7_alt_recurrent_kernel<<<grid, block, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      r.data_ptr<float>(), w.data_ptr<float>(), k.data_ptr<float>(),
      v.data_ptr<float>(), kk.data_ptr<float>(), a.data_ptr<float>(),
      initial_state_ptr, out.data_ptr<float>(), final_state.data_ptr<float>(),
      batch_size, seq_len, num_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out, final_state};
}
