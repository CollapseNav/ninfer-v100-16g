#pragma once

// ninfer::ops - fused rmsnorm + split fold-rotation launcher. Host-compiled; never includes the
// kernel header.
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// x [k, tokens] bf16 in, norm_weight [k], h [k, tokens] bf16 norm output, folded [rotation_k,
// tokens] fp16 rotation output. signs/n_blk/rotation_k describe the Hadamard block, exactly as the
// rotation launcher takes them. Decode-shaped calls only; the wrapper checks.
void rmsnorm_rotate_launch(const Tensor& x, const Tensor& norm_weight, float eps, const Tensor& h,
                           Tensor& folded, const float* signs, int n_blk, int rotation_k,
                           cudaStream_t stream);

} // namespace ninfer::ops::detail
