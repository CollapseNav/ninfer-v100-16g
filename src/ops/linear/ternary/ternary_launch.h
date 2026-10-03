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

// Fused-residual arm of the PQ2 GEMV: the same kernel as the shipped one, with the layer's
// residual add folded into its epilogue so that `out` is both read and written and no separate
// ops::residual_add launch is needed. Bit-identical to the composed route -- see gemv_store().
//
// `admits` decides the GEOMETRY, the token band and the route knobs; `x` may be the raw activation
// in either container. The container the fused kernel needs (fp16) is decided by
// ternary_activation_is_fp16() before the rotation, because it is the same predicate
// folded_activation() applies -- and launch_ternary_pq2_gemv_add() rejects a non-fp16 activation
// outright, so the two can never disagree silently.
//
// Band: the shipped decode/verify arms -- the staged kernel at T = 1..2 (staging on at its shipped
// depth) and the small-tile kernel at T = 3..5, with the default route knobs. Everything else
// (prefill, the QPN verify band, a switched arm) returns false and the caller composes the GEMM and
// ops::residual_add as before.
[[nodiscard]] bool ternary_pq2_gemv_add_admits(const Tensor& x, const Weight& w) noexcept;
void launch_ternary_pq2_gemv_add(const Tensor& x, const Weight& w, Tensor& out,
                                 std::int32_t out_row_stride, cudaStream_t stream);

} // namespace ninfer::ops::detail
