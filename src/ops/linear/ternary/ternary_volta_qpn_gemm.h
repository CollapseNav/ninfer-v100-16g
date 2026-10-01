#pragma once

// Host-side entry points for the PQ2 Volta tensor-core (QPN) route. See
// ternary_volta_qpn_gemm.cuh for the schedule; this header stays device-free so the dispatcher can
// include it without pulling the mma PTX into a host translation unit.

#include "core/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// NINFER_TERNARY_QPN=0 disables the route (it is an experiment while it is being measured); unset
// means enabled wherever the shape qualifies.
[[nodiscard]] bool ternary_qpn_enabled() noexcept;

// n = output rows, k = reduction width, t = live tokens.
[[nodiscard]] bool ternary_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                               std::int32_t t) noexcept;

// Permutes a ternary weight code+scale planes in place into the QPN tile order
// [n/32][group][lane][32B code][2B scale]. Load-time only and irreversible: after this the SIMT
// kernels must not touch the weight, which the dispatcher's prepacked guard enforces.
void ternary_prepack_qpn(Weight& w, cudaStream_t stream = nullptr);

[[nodiscard]] bool ternary_qpn_is_prepacked(const void* qdata) noexcept;

// x is the folded activation [k, t] (fp16 or bf16), out is [n, t] with the given row stride.
void launch_ternary_volta_qpn(const Tensor& x_folded, const Weight& w, Tensor& out,
                              std::int32_t out_row_stride, cudaStream_t stream);

} // namespace ninfer::ops::detail
