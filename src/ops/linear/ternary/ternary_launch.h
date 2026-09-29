// Ported to the Volta (sm_70) tree from the Ambolio/ninfer-4090-windows ternary lineage.
// The original banner read "Ada sm_89"; this copy is built and verified for sm_70 only.
// See docs/ternary-port.md for what was changed and what was deliberately left out.
#pragma once

#include "core/tensor.h"
#include "ops/linear/ternary/ternary_s8_scratch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The token tile is the only schedule knob on the reference path: one token per CTA at
// T = 1 (decode), eight otherwise (a small prefill). Both share one kernel template.
//
// out_row_stride is the row count of the OUT tensor's parent allocation (the token stride), so a
// caller can have several weights write disjoint row ranges of one fused output -- the split GDN
// and attention parents do exactly that. Pass w.n when the output is the whole tensor.
//
// The int8 rung (s8) needs an activation-quantization scratch buffer, which the caller owns because
// this op runs inside captured CUDA graphs (a lazy cudaMalloc at launch time would be illegal).
// Callers without a workspace -- the *_basis entry points used by the fused GDN/attention parents --
// pass an empty scratch and therefore stay on the bf16 rungs.
using TernaryLaunch = void (*)(const Tensor&, const Weight&, Tensor&, std::int32_t, cudaStream_t,
                               TernaryS8Scratch);

void launch_ternary_gemm_t1(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch scratch);
void launch_ternary_gemm_t8(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch scratch);

// Volta (sm_70) fp16 tensor-core prefill route, in ternary_volta_mma_gemm.cu. Deliberately not
// part of TernaryLaunch: it needs no scratch buffer, and it is a whole-GEMM entry point rather
// than one of the token-tile schedules select_ternary_launch() picks between.
//
// `admits` returns false -- and the caller must then fall through to the SIMT routes -- whenever
// the shape does not qualify, including every case where the split-K count is above one, because
// the split-K path needs an fp32 partial workspace that this op has no arena for. See the .cu.
[[nodiscard]] bool ternary_volta_mma_admits(const Tensor& x, const Weight& w) noexcept;
void launch_ternary_volta_mma(const Tensor& x, const Weight& w, Tensor& out,
                              std::int32_t out_row_stride, cudaStream_t stream);

} // namespace ninfer::ops::detail
