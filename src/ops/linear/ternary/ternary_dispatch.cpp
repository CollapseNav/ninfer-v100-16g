// Ported to the Volta (sm_70) tree from the Ambolio/ninfer-4090-windows ternary lineage.
// The original banner read "Ada sm_89"; this copy is built and verified for sm_70 only.
//
// One change from the original: the int8 rung's scratch allocation is compiled out under
// NINFER_VOLTA_BUILD. That rung is a tensor-core kernel (m16n8k32.s8), so on sm_70 the launcher
// ignores the scratch entirely; leaving the allocation in would ask the arena for up to
// k*T bytes plus 4*T that nothing ever reads, and the arena is exactly what the ternary linear op
// runs out of first. The guard is here rather than in the launcher because this file is the only
// thing that knows the token threshold.
#include "ops/linear/ternary/ternary_dispatch.h"

#include "ops/linear/ternary/ternary_cutlass_sm70.h"
#include "ops/linear/ternary/ternary_rotation.h"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

TernaryLaunch select_ternary_launch(std::int32_t n, std::int32_t k, std::int32_t t,
                                    LinearPolicy policy) {
    if (n <= 0 || k <= 0 || t <= 0) {
        throw std::invalid_argument("ternary linear: unsupported shape or T");
    }
    // The reference kernel decodes whole 128-weight groups, so K has to be a whole number
    // of them. Every width the model uses (5120, 6144, 10240, 17408) satisfies this, and
    // row_split_geometry() pads to the same 128 boundary, so no padding groups exist.
    if (k % PTQ1RowSplitStorage::kGroupK != 0) {
        throw std::invalid_argument("ternary linear: K must be a multiple of the group size");
    }
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    // Both ternary formats take the same schedule; the atom is chosen from w.qtype inside
    // the launch. Decode (T = 1) and small prefill use different token tiles.
    return t == 1 ? launch_ternary_gemm_t1 : launch_ternary_gemm_t8;
}

void ternary_dispatch_basis_strided(const Tensor& x_folded, const Weight& w, Tensor& out,
                                    std::int32_t out_row_stride, LinearPolicy policy,
                                    WorkspaceArena* workspace, cudaStream_t stream) {
    const TernaryLaunch launch = select_ternary_launch(w.n, w.k, x_folded.ne[1], policy);
    // The CUTLASS arm needs a scratch buffer it cannot take from TernaryLaunch, which carries none,
    // so it is selected here and only when the CALLER was able to hand an arena in. The parents that
    // feed several folded weights from one activation do carry one (they had to, for the rotation),
    // and the workspace-free callers simply stay on the fused routes. The activation is not consumed:
    // the launcher takes its own fp16 copy precisely so that a shared activation survives the four
    // or three projections that read it. See ternary_cutlass_sm70.h.
    if (workspace != nullptr && ternary_cutlass_sm70_admits(w, x_folded.ne[1])) {
        ternary_cutlass_sm70_launch(x_folded, w, out, out_row_stride, *workspace, stream);
        return;
    }
    // The caller already folded the activation, and may hand in the int8 scratch it allocated from
    // its own arena. With an empty scratch these calls stay on the bf16 rungs, which is what they
    // did before the s8 rung existed.
    launch(x_folded, w, out, out_row_stride, stream, TernaryS8Scratch{});
}

void ternary_dispatch_basis(const Tensor& x_folded, const Weight& w, Tensor& out,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    ternary_dispatch_basis_strided(x_folded, w, out, w.n, policy, workspace, stream);
}

void ternary_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                      WorkspaceArena* workspace, cudaStream_t stream) {
    const TernaryLaunch launch = select_ternary_launch(w.n, w.k, x.ne[1], policy);

    if (!ternary_rotation_enabled()) {
        // NINFER_TERNARY_HADAMARD=0: run the GEMM against the raw activation so the full forward
        // pass (and therefore the decode speed) is measurable. The result is numerically
        // meaningless -- the activation is in the wrong basis -- which is the point: it separates
        // "the ternary decode is broken" from "the rotation is broken" without a rebuild.
        launch(x, w, out, w.n, stream, TernaryS8Scratch{});
        return;
    }
    if (workspace == nullptr) {
        // Name the shape in the message: a folded ternary weight reaching the workspace-free
        // entry point is a call-site wiring gap, and the shape is what identifies the weight.
        throw std::invalid_argument(
            "ternary linear: the folded basis needs a rotation workspace [N=" +
            std::to_string(w.n) + ", K=" + std::to_string(w.k) + ", T=" +
            std::to_string(x.ne[1]) + ", qtype=" + std::to_string(static_cast<int>(w.qtype)) + "]");
    }

    // Scoped: the scratch is handed back when this op returns, so it does not accumulate across
    // the (many) graph constructions of one load. Sequential reuse on one stream is safe.
    // folded_activation() rejects a ternary weight with no sign block, so an unfolded artifact
    // fails loudly here instead of silently multiplying by unfolded weights.
    auto scope              = workspace->scope();
    const Tensor activation = folded_activation(x, w, *workspace, stream);

    // The CUTLASS arm sits ahead of the token-tile schedules because it is a whole-GEMM entry point
    // rather than one of them: it dequantises the weight chunk to fp16 once and then runs the
    // CUTLASS sm70 tensor-op GEMM, which measures 94-100 TFLOP/s on this model's shapes against the
    // fused kernel's 36.7. It refuses every shape it cannot serve -- most importantly every token
    // count below its threshold, which is what keeps decode off this path -- and those fall through
    // to `launch` below. It takes its own fp16 copy of the activation; see the header.
    if (ternary_cutlass_sm70_admits(w, activation.ne[1])) {
        ternary_cutlass_sm70_launch(activation, w, out, w.n, *workspace, stream);
        return;
    }

#ifndef NINFER_VOLTA_BUILD
    // int8 rung scratch: one int8 code row per token plus one fp32 scale per token. Taken from the
    // same arena the rotation just used, and counted by ternary_rotation_workspace_bytes() so the
    // planner sizes the arena for it -- allocating it lazily inside the launch would be illegal,
    // because this op runs inside captured CUDA graphs. Below the threshold the scratch is left
    // empty and the launch falls through to the bf16 rungs.
    TernaryS8Scratch scratch{};
    if (x.ne[1] >= kTernaryS8MinTokens) {
        const DeviceSpan codes  = workspace->alloc_bytes(ternary_s8_codes_bytes(w.k, x.ne[1]));
        const DeviceSpan scales = workspace->alloc_bytes(ternary_s8_scales_bytes(x.ne[1]));
        scratch.codes           = static_cast<std::int8_t*>(codes.data);
        scratch.scales          = static_cast<float*>(scales.data);
    }
    launch(activation, w, out, w.n, stream, scratch);
#else
    // sm_70: the s8 rung is a tensor-core kernel that does not exist here, and the launcher
    // ignores the scratch. Do not allocate it -- see the file header.
    launch(activation, w, out, w.n, stream, TernaryS8Scratch{});
#endif
}

} // namespace ninfer::ops::detail
