// Ported to the Volta (sm_70) tree from the Ambolio/ninfer-4090-windows ternary lineage.
// The original banner read "Ada sm_89"; this copy is built and verified for sm_70 only.
// See docs/ternary-port.md for what was changed and what was deliberately left out.
#pragma once

#include "core/arena.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/ternary/ternary_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

TernaryLaunch select_ternary_launch(std::int32_t n, std::int32_t k, std::int32_t t,
                                    LinearPolicy policy);

// Rotate the activation into the folded basis when the weight needs it, then run the GEMM.
void ternary_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                      WorkspaceArena* workspace, cudaStream_t stream);

// Run the GEMM when the activation is ALREADY in the folded basis. Callers that feed several
// folded weights from one activation (the split GDN input projection feeds three) rotate once and
// then call this, instead of paying for -- and diverging on -- a redundant rotation per weight.
//
// `workspace` is the arena the CUTLASS prefill arm takes its dequantised weight chunk from. It may
// be null -- the workspace-free callers then stay on the fused routes -- but a caller that HAS an
// arena should pass it, or that parent silently keeps its projections on the slow route.
void ternary_dispatch_basis(const Tensor& x_folded, const Weight& w, Tensor& out,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

// Same, for a weight that writes a ROW RANGE of a larger fused output. `out` must already point at
// the range start (Tensor::slice over ne[0], the contiguous axis, does that) and out_row_stride is
// the parent's row count -- which is also the token stride under ninfer's ne[0]-contiguous layout.
void ternary_dispatch_basis_strided(const Tensor& x_folded, const Weight& w, Tensor& out,
                                    std::int32_t out_row_stride, LinearPolicy policy,
                                    WorkspaceArena* workspace, cudaStream_t stream);

// residual_out += W * x, with the add folded into the GEMM's epilogue when the decode/verify arm
// can carry it. Returns FALSE without launching anything when it cannot -- the caller must then
// compose: ternary_dispatch() into a scratch, then ops::residual_add(). Every non-default route
// (prefill, the QPN verify band, any switched arm, rotation disabled) takes that path.
//
// Bit-identical to the composed route on both paths: the fused epilogue rounds the projection to
// bf16 before adding, which is what the composed route's intermediate scratch does. See
// gemv_store() in ternary_rowsplit_gemv.cuh.
[[nodiscard]] bool ternary_dispatch_add(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        LinearPolicy policy, WorkspaceArena* workspace,
                                        cudaStream_t stream);

} // namespace ninfer::ops::detail
