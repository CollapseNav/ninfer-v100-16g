#pragma once

// Host-side entry points for the PQ2 Volta tensor-core (QPN) route. See
// ternary_volta_qpn_gemm.cuh for the schedule; this header stays device-free so the dispatcher can
// include it without pulling the mma PTX into a host translation unit.

#include "core/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// NINFER_TERNARY_QPN=0 disables the route; unset means enabled wherever the shape and the token
// count qualify.
[[nodiscard]] bool ternary_qpn_enabled() noexcept;

// n = output rows, k = reduction width, t = live tokens. True for 6 <= t <= 32 (the wide-verify
// band; the T = 1..5 decode and MTP windows stay on the SIMT tile kernel, where the route loses).
[[nodiscard]] bool ternary_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                               std::int32_t t) noexcept;

// Permutes a ternary weight code+scale planes in place into the QPN tile order
// [n/32][group][lane][32B code][2B scale]. Load-time only and irreversible: after this the SIMT
// kernels must not touch the weight, which the dispatcher's prepacked guard enforces.
//
// PARKED AND UNCALLED. The QPN kernel reads the artifact's row-major plane (see the reader comment
// in the .cuh): a lane owns an output row and that row's 32-byte group is contiguous, so two uint4
// loads already fetch one fully-used sector per lane, and the permutation is worth nothing that
// vectorising the read does not. Enabling it would require the fused MMA (T = 32..255) and the
// CUTLASS dequant pass (T >= 256) to learn the permuted index before prefill would run at all.
// Kept as the record of that alternative; nothing calls it.
void ternary_prepack_qpn(Weight& w, cudaStream_t stream = nullptr);

[[nodiscard]] bool ternary_qpn_is_prepacked(const void* qdata) noexcept;

// x is the folded activation [k, t] (fp16 or bf16), out is [n, t] with the given row stride.
void launch_ternary_volta_qpn(const Tensor& x_folded, const Weight& w, Tensor& out,
                              std::int32_t out_row_stride, cudaStream_t stream);

} // namespace ninfer::ops::detail
