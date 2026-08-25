#include <torch/all.h>
#include <torch/library.h>

#include "core/registration.h"
#include "ops.h"

TORCH_LIBRARY_FRAGMENT(_C, rwkv7_ops) {
  rwkv7_ops.def(
      "rwkv7_alt_recurrent("
      "Tensor r, Tensor w, Tensor k, Tensor v, Tensor kk, Tensor a, "
      "Tensor? initial_state=None) -> (Tensor, Tensor)");
  rwkv7_ops.def(
      "rwkv7_reduce_d64_atten_exact(Tensor state, Tensor vector) -> Tensor");
  rwkv7_ops.def(
      "rwkv7_recurrent_t1_exact_update("
      "Tensor state, Tensor exp_w, Tensor kk_a, Tensor k, Tensor v, "
      "Tensor sa) -> Tensor");
  rwkv7_ops.def(
      "rwkv7_recurrent_t1_exact_direct_cache("
      "Tensor(a!) cache, Tensor slot_ids, Tensor exp_w, Tensor kk, "
      "Tensor kk_a, Tensor k, Tensor v, Tensor r) -> Tensor");
  rwkv7_ops.def(
      "rwkv7_masked_store(Tensor(a!) cache, Tensor values, Tensor slot_ids) "
      "-> ()");
  rwkv7_ops.def("rwkv7_strided_gather(Tensor cache, Tensor slot_ids) -> Tensor");
}

TORCH_LIBRARY_IMPL(_C, CUDA, rwkv7_ops) {
  rwkv7_ops.impl("rwkv7_alt_recurrent", &rwkv7_alt_recurrent);
  rwkv7_ops.impl("rwkv7_reduce_d64_atten_exact",
                  &rwkv7_reduce_d64_atten_exact);
  rwkv7_ops.impl("rwkv7_recurrent_t1_exact_update",
                  &rwkv7_recurrent_t1_exact_update);
  rwkv7_ops.impl("rwkv7_recurrent_t1_exact_direct_cache",
                  &rwkv7_recurrent_t1_exact_direct_cache);
  rwkv7_ops.impl("rwkv7_masked_store", &rwkv7_masked_store);
  rwkv7_ops.impl("rwkv7_strided_gather", &rwkv7_strided_gather);
}

REGISTER_EXTENSION(_C);
