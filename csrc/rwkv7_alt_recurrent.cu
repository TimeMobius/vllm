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

// Mirrors the ATen CUDA reduce configuration used by
// (state * vector.unsqueeze(-1)).sum(dim=-2) for [B, H, 64, 64] FP32 state:
// output-vector size 4, 32x4 CTA, four independent accumulators over D.
// The narrow shape/dtype guard is deliberate: this preserves the exact
// reduction contract established by the corresponding RWKV7 integration tests.
__global__ void rwkv7_reduce_d64_atten_exact_kernel(
    const float* __restrict__ state, const float* __restrict__ vector,
    float* __restrict__ out, const int64_t output_numel) {
  // Exact launch/dataflow mirror of the observed ATen kernel:
  // reduce_kernel<128, 4, ReduceOp<float, sum, ..., 4, 4>> with a 32x4 CTA.
  // Each CTA owns 32 output vectors (four adjacent output values each); the
  // four y-lanes cooperatively reduce D=64 using the same four independent
  // accumulators and the same 4 -> 2 -> 1 shared-memory combine order.
  constexpr int kOutputVector = 4;
  constexpr int kHeadDim = 64;
  constexpr int kThreadsX = 32;
  constexpr int kThreadsY = 4;
  __shared__ float shared[kThreadsY][kThreadsX][kOutputVector];

  const int64_t output_base =
      (static_cast<int64_t>(blockIdx.x) * kThreadsX + threadIdx.x) *
      kOutputVector;
  if (output_base >= output_numel) {
    return;
  }
  const int64_t bh = output_base / kHeadDim;
  const int64_t v_base = output_base - bh * kHeadDim;
  const float* state_base = state + bh * kHeadDim * kHeadDim + v_base;
  const float* vector_base = vector + bh * kHeadDim;

  float accum[kOutputVector][kThreadsY] = {};
#pragma unroll
  for (int d_base = threadIdx.y; d_base < kHeadDim; d_base += 16) {
#pragma unroll
    for (int i = 0; i < kThreadsY; ++i) {
#pragma unroll
      for (int lane = 0; lane < kOutputVector; ++lane) {
        const int d = d_base + i * kThreadsY;
        accum[lane][i] = __fadd_rn(
            accum[lane][i],
            __fmul_rn(state_base[d * kHeadDim + lane], vector_base[d]));
      }
    }
  }

#pragma unroll
  for (int lane = 0; lane < kOutputVector; ++lane) {
    float combined = __fadd_rn(accum[lane][0], accum[lane][1]);
    combined = __fadd_rn(combined, accum[lane][2]);
    shared[threadIdx.y][threadIdx.x][lane] =
        __fadd_rn(combined, accum[lane][3]);
  }
  __syncthreads();
  if (threadIdx.y < 2) {
#pragma unroll
    for (int lane = 0; lane < kOutputVector; ++lane) {
      shared[threadIdx.y][threadIdx.x][lane] =
          __fadd_rn(shared[threadIdx.y][threadIdx.x][lane],
                    shared[threadIdx.y + 2][threadIdx.x][lane]);
    }
  }
  __syncthreads();
  if (threadIdx.y == 0) {
#pragma unroll
    for (int lane = 0; lane < kOutputVector; ++lane) {
      out[output_base + lane] =
          __fadd_rn(shared[0][threadIdx.x][lane], shared[1][threadIdx.x][lane]);
    }
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

__global__ void rwkv7_recurrent_t1_exact_update_kernel(
    const float* __restrict__ state, const float* __restrict__ exp_w,
    const float* __restrict__ kk_a, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ sa,
    float* __restrict__ out, const int64_t numel) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }

  // state is [B, H, 64, 64]. Each scalar term is [B, H, 64] and each
  // vector term is [B, H, 64]. Explicit round-to-nearest operations reproduce
  // eager ATen's materialized product/add order while avoiding its five large
  // intermediate state tensors.
  const int64_t value_idx = idx & (kRWKV7AltHeadDim - 1);
  const int64_t scalar_idx = idx >> 6;
  const int64_t batch_head_idx = scalar_idx >> 6;
  volatile float state_decay = __fmul_rn(exp_w[scalar_idx], state[idx]);
  volatile float key_sa = __fmul_rn(
      kk_a[scalar_idx], sa[batch_head_idx * kRWKV7AltHeadDim + value_idx]);
  volatile float key_value = __fmul_rn(
      k[scalar_idx], v[batch_head_idx * kRWKV7AltHeadDim + value_idx]);
  volatile float partial = __fadd_rn(state_decay, key_sa);
  out[idx] = __fadd_rn(partial, key_value);
}

void check_rwkv7_recurrent_t1_exact_update_tensor(
    const torch::Tensor& tensor, const char* name,
    const std::vector<int64_t>& expected_shape) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name,
              " must have dtype float32.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.sizes().vec() == expected_shape, name, " must have shape ",
              expected_shape, ", got ", tensor.sizes(), ".");
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

torch::Tensor rwkv7_reduce_d64_atten_exact(const torch::Tensor& state,
                                           const torch::Tensor& vector) {
  TORCH_CHECK(state.is_cuda() && vector.is_cuda(),
              "`state` and `vector` must be CUDA tensors.");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32 &&
                  vector.scalar_type() == torch::kFloat32,
              "`state` and `vector` must use float32.");
  TORCH_CHECK(state.is_contiguous() && vector.is_contiguous(),
              "`state` and `vector` must be contiguous.");
  TORCH_CHECK(state.dim() == 4 && state.size(2) == kRWKV7AltHeadDim &&
                  state.size(3) == kRWKV7AltHeadDim,
              "`state` must have shape [B, H, 64, 64].");
  TORCH_CHECK(vector.sizes() == state.sizes().slice(0, 3),
              "`vector` must have shape [B, H, 64].");
  TORCH_CHECK(state.device() == vector.device(),
              "`state` and `vector` must share a CUDA device.");
  c10::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(state.get_device());
  auto out = torch::empty({state.size(0), state.size(1), kRWKV7AltHeadDim},
                          state.options());
  constexpr int kThreadsX = 32;
  constexpr int kThreadsY = 4;
  constexpr int kOutputVector = 4;
  const int64_t outputs_per_block = kThreadsX * kOutputVector;
  const int64_t blocks =
      (out.numel() + outputs_per_block - 1) / outputs_per_block;
  rwkv7_reduce_d64_atten_exact_kernel<<<blocks, dim3(kThreadsX, kThreadsY), 0,
                                        at::cuda::getCurrentCUDAStream()>>>(
      state.data_ptr<float>(), vector.data_ptr<float>(), out.data_ptr<float>(),
      out.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor rwkv7_recurrent_t1_exact_update(const torch::Tensor& state,
                                              const torch::Tensor& exp_w,
                                              const torch::Tensor& kk_a,
                                              const torch::Tensor& k,
                                              const torch::Tensor& v,
                                              const torch::Tensor& sa) {
  TORCH_CHECK(state.dim() == 4 && state.size(2) == kRWKV7AltHeadDim &&
                  state.size(3) == kRWKV7AltHeadDim,
              "`state` must have shape [B, H, 64, 64], got ", state.sizes(),
              ".");
  const int64_t batch_size = state.size(0);
  const int64_t num_heads = state.size(1);
  const std::vector<int64_t> scalar_shape = {batch_size, num_heads,
                                             kRWKV7AltHeadDim};
  check_rwkv7_recurrent_t1_exact_update_tensor(
      state, "`state`",
      {batch_size, num_heads, kRWKV7AltHeadDim, kRWKV7AltHeadDim});
  check_rwkv7_recurrent_t1_exact_update_tensor(exp_w, "`exp_w`", scalar_shape);
  check_rwkv7_recurrent_t1_exact_update_tensor(kk_a, "`kk_a`", scalar_shape);
  check_rwkv7_recurrent_t1_exact_update_tensor(k, "`k`", scalar_shape);
  check_rwkv7_recurrent_t1_exact_update_tensor(v, "`v`", scalar_shape);
  check_rwkv7_recurrent_t1_exact_update_tensor(sa, "`sa`", scalar_shape);
  TORCH_CHECK(exp_w.device() == state.device() &&
                  kk_a.device() == state.device() &&
                  k.device() == state.device() &&
                  v.device() == state.device() && sa.device() == state.device(),
              "all recurrent update tensors must share `state`'s CUDA device.");

  c10::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(state.get_device());
  auto out = torch::empty_like(state);
  constexpr int kThreads = 256;
  const int64_t numel = state.numel();
  rwkv7_recurrent_t1_exact_update_kernel<<<(numel + kThreads - 1) / kThreads,
                                           kThreads, 0,
                                           at::cuda::getCurrentCUDAStream()>>>(
      state.data_ptr<float>(), exp_w.data_ptr<float>(), kk_a.data_ptr<float>(),
      k.data_ptr<float>(), v.data_ptr<float>(), sa.data_ptr<float>(),
      out.data_ptr<float>(), numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
