// Route 3 for the ternary prefill: dequantise the PQ2_0_G128 weight chunk to plain fp16 once, then
// hand the multiply-accumulate to the CUTLASS sm70 fp16 tensor-op GEMM that the fp8 path already
// uses in this tree.
//
// Why this exists. The fused PQ2 tensor-core kernel (ternary_volta_mma_gemm) decodes the code word
// inside the CTA and feeds the MMA from shared memory. Its measured ceiling is set by shared-memory
// traffic, not by the tensor core: per k-slice the A-fragment load (4 cycles) and the decode's
// shared store (4 cycles) bracket 4 cycles of MMA, so ~28-34% of the fp16 peak is all the shape can
// reach -- 36.7 TFLOP/s measured. CUTLASS's own pipeline has no such constraint. Measured with
// tools/v100/ternary-probes/cutlass_probe.cu on this card, on this model's exact shapes with plain fp16 operands:
//
//     qkv/gate_up 3412x17408x5120   99.7 TFLOP/s   79.7% of the 125 TFLOP/s peak
//     o/down      3412x 5120x17408  93.8 TFLOP/s   75.1%
//     lm_head     3412x248320x5120  99.3 TFLOP/s   79.4%
//
// so the route trades 2.4x more compute for 8x more weight traffic: PQ2 is 0.266 B/weight and fp16
// is 2 B/weight, and the GEMM re-reads the B panel once per M-tile. It is therefore a PREFILL-ONLY
// route. At M=1 the same GEMM measures 0.3 TFLOP/s, so decode must never come here; admits() gates
// on the token count.
//
// The dequantised chunk is bounded to kMaxChunkBytes so the arena reservation stays small: the
// largest single linear in this model is 5120x17408, which would otherwise ask for 178 MB.
#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cstdint>

namespace ninfer::ops::detail {

// Called from the workspace planner. Must agree exactly with what launch() allocates, and must
// return 0 for every shape admits() refuses -- otherwise the planner reserves for a route that is
// never taken (and, worse, the reverse would allocate inside a captured graph).
//
// It deliberately takes K and the token count only, never N: the attention and GDN parents are
// planned from (input_rows, max_tokens) alone, and a reservation that needed N could not be stated
// there. The launcher allocates a strict subset of this, so the direction of the approximation is
// always safe.
[[nodiscard]] std::size_t ternary_cutlass_sm70_workspace_bytes(std::int32_t k, std::int32_t cols);

[[nodiscard]] bool ternary_cutlass_sm70_admits(const Weight& w, std::int32_t tokens) noexcept;

// `x_folded` is the activation already rotated into the folded basis. The launcher allocates its own
// fp16 copy of it in `ws` and never writes through `x_folded`, because the attention and GDN parents
// feed four and three folded weights from ONE activation and an in-place conversion would corrupt
// the projections that follow. Do not reuse x_folded after this call only in the sense that the
// caller must keep it alive until the stream drains.
void ternary_cutlass_sm70_launch(const Tensor& x_folded, const Weight& w, Tensor& out,
                                 std::int32_t out_row_stride, WorkspaceArena& ws,
                                 cudaStream_t stream);

} // namespace ninfer::ops::detail
