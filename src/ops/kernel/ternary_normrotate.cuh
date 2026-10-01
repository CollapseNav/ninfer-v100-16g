#pragma once

// Fused rmsnorm + split fold-rotation: one launch where the graph would run rmsnorm(...) and then
// the consumer's folded_activation(...). Decode only -- the caller gates on the same token bound as
// NINFER_TERNARY_ROTATE_SPLIT, so the folded output stays fp16 exactly where folded_activation()
// would have produced fp16, and prefill keeps the two-launch path it uses today.
//
// Both halves are bit-identical to what they replace:
//   phase 1 is the shipped rmsnorm_cta_bf16x2_kernel<Offset, 256, 10, true, 5120> body, verbatim --
//   same loads, same block_reduce_sum, same Offset epilogue, same bf16 store into h;
//   phase 2 runs split_transform_unit(), the same function the split kernel runs, on those exact
//   bf16 values, with one four-warp group per D1024 and the same stage order.
// All eight warps participate in every round (the short last round clamps to the final unit and
// rewrites it with identical values) so the __syncthreads sequence stays uniform.
//
// What this buys: one launch floor (~2.6 us) per call instead of two, at 128 calls per decode step.

#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear/ternary/ternary_rotation_split.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <RmsEpilogue Epilogue, int Block, int MaxPairsPerThread, bool Prefetch, int FixedD,
          bool kFp16Out>
__launch_bounds__(Block) __global__ void rmsnorm_rotate_kernel(
    const __nv_bfloat162* x, const __nv_bfloat162* weight, const __nv_bfloat162* z, __nv_bfloat162* h,
    void* folded, const float* signs, int n_blk, int k, int tokens, float eps) {
    static_assert(Block % kWarpSize == 0);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= tokens) { return; }

    // ---- phase 1: verbatim rmsnorm_cta body (FixedD row, one row per CTA) --------------------
    // Phase 1 always computes as the shipped 256-thread norm: indexing, pair count and the
    // reduction are pinned to kNormThreads so a wider launch (Block=640, giving phase 2 all
    // five units in one round) only changes which threads are idle. Idle threads accumulate
    // 0, write unread slots of warp_sums (sized from Block), pass the same __syncthreads and
    // skip the stores -- and block_reduce_sum<256> reads only slots 0..7, i.e. exactly the
    // 256 active threads, so the value and its rounding tree are bit-identical to the
    // 256-thread launch.
    constexpr int kNormThreads  = 256;
    const int d                 = FixedD;
    const int pairs             = d / 2;
    const int pairs_per_thread  = pairs / kNormThreads;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[MaxPairsPerThread];
    __nv_bfloat162 weights[MaxPairsPerThread];
    __nv_bfloat162 gates[MaxPairsPerThread];
    float sum = 0.0f;

#pragma unroll
    for (int pi = 0; pi < MaxPairsPerThread; ++pi) {
        if (threadIdx.x < kNormThreads && pi < pairs_per_thread) {
            const int pair = static_cast<int>(threadIdx.x) + pi * kNormThreads;
            values[pi]     = x[row_base + pair];
            if constexpr (Prefetch) {
                weights[pi] = weight[pair];
                if constexpr (Epilogue == RmsEpilogue::Gated) { gates[pi] = z[row_base + pair]; }
            }
            const float2 xf = __bfloat1622float2(values[pi]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    __shared__ float warp_sums[Block / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<kNormThreads>(sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(d) + eps); }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int pi = 0; pi < MaxPairsPerThread; ++pi) {
        if (threadIdx.x < kNormThreads && pi < pairs_per_thread) {
            const int pair  = static_cast<int>(threadIdx.x) + pi * kNormThreads;
            const float2 xf = __bfloat1622float2(values[pi]);
            __nv_bfloat162 w_pair;
            __nv_bfloat162 z_pair{};
            if constexpr (Prefetch) {
                w_pair = weights[pi];
                if constexpr (Epilogue == RmsEpilogue::Gated) { z_pair = gates[pi]; }
            } else {
                w_pair = weight[pair];
                if constexpr (Epilogue == RmsEpilogue::Gated) { z_pair = z[row_base + pair]; }
            }
            const float2 wf = __bfloat1622float2(w_pair);
            float2 zf{0.0f, 0.0f};
            if constexpr (Epilogue == RmsEpilogue::Gated) { zf = __bfloat1622float2(z_pair); }
            h[row_base + pair] =
                __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        }
    }

    // ---- phase 2: the split transform, one four-warp group per D1024 unit --------------------
    __syncthreads(); // every h store above is visible to every thread of this block
    constexpr int kUnits  = FixedD >> 10;
    constexpr int kGroups = Block / (kSplitWarps * kWarpSize); // 2 at 256, 5 at 640
    __shared__ float exchange[kGroups][kSplitWarps][32][kSplitElems];
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int group = warp / kSplitWarps;
    const int role  = warp & (kSplitWarps - 1);
    const int lane  = static_cast<int>(threadIdx.x) & (kThreadsPerWarp - 1);
    for (int u0 = 0; u0 < kUnits; u0 += kGroups) {
        const int unit_idx = (u0 + group < kUnits) ? (u0 + group) : (kUnits - 1);
        const std::int64_t base = row * static_cast<std::int64_t>(k) +
                                  (static_cast<std::int64_t>(unit_idx) << 10) +
                                  (static_cast<std::int64_t>(role) << 8) + (lane << 3);
        split_transform_unit<kFp16Out>(
            reinterpret_cast<const __nv_bfloat16*>(h) + base, folded,
            signs + static_cast<std::int64_t>(unit_idx % n_blk) * kBlockSize + (role << 8) +
                (lane << 3),
            exchange[group], lane, role, base);
        __syncthreads();
    }
}

} // namespace
} // namespace ninfer::ops::detail
