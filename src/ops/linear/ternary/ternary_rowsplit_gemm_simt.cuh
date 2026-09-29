// Ported to the Volta (sm_70) tree for the ternary PQ2_0_G128 port.
#pragma once

// PQ2_0_G128 RowSplit x BF16 SIMT GEMM.
//
// out[Rows, Cols] = W[Rows, K] * x[K, Cols]
//
// One warp owns one output row across ColsPerTile columns. Raw PQ2 code words and adjacent FP16
// scale pairs are staged per row with cp.async; each group's sixteen decoded FP32 weights are
// reused across the whole column tile and accumulated with FP32 FMA.
//
// Structurally this is v100's q4_rowsplit_gemm_simt.cuh with the group geometry swapped. Q4 packs
// eight 4-bit codes per 32-byte group; PQ2 packs sixteen 2-bit codes into the same 32 bytes. The
// staging geometry is therefore bit-identical -- kCodeVecsPerStage is kGroupsPerStage * 2 either
// way, and the scale plane is one binary16 per group in both -- and only two numbers change: the
// per-phase K offset and the per-lane element count, from `phase * 256 + lane * 8` to
// `phase * 512 + lane * 16`. The two bits that differ are also exactly the two that make the
// code plane whole 32-bit words, which is why this kernel serves PQ2_0 only and PTQ1_0 stays on
// the reference kernel.
//
// Why it exists: sm_70 has no bf16 or s8 tensor-core schedule, so the author's tree leaves every
// T >= 5 on the row-blocked GEMV, which its own profiling calls issue-bound with no occupancy left
// to buy (95.47% achieved occupancy, ALU the top pipe, 909e6 instructions against a 1.32 ms
// pure-issue floor). The structural difference here is that every warp in the CTA owns a DIFFERENT
// output row but reads the SAME activation slice, so the stage is copied to shared memory once per
// CTA instead of once per warp per column. The blocked GEMV cannot afford that at its register
// budget, which is precisely why it lost on the author's card.
//
// Deliberately NOT attempted: the multiply-free "signed activation sum" that the T2 port uses.
// PQ2 codes are (code - 1) in {-1, 0, +1}, so a group's partial could be a masked sum rather than
// sixteen FMAs -- but the FMA count per WEIGHT is already 1 for both Q4 and this kernel (sixteen
// weights over the sixteen elements one lane owns), so the masked form trades sixteen FMAs for
// roughly two selects plus an add per element, which is not obviously cheaper. v100's own notes
// also record that the HFMA2 and dp4a variants of this kernel family were measured and rejected.
// FP32 FMA matches the proven choice; the multiply-free form is an open experiment, not a win.

#include "core/pdl.cuh"
#include "ops/common/math.cuh" // bf16x2_bits_to_float2; the bonsai storage header pulls in no
                               // project headers at all, unlike q4_rowsplit_storage.cuh
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Storage_, class Atom_, int RowsPerCta_, int ColsPerTile_, int GroupsPerStage_,
          int PipelineStages_, Cache CodeCache_, int LaunchBoundsMinBlocks_>
struct TernaryRowSplitSimtGemmSchedule {
    using Storage = Storage_;
    using Atom    = Atom_;

    static constexpr int kRowsPerCta            = RowsPerCta_;
    static constexpr int kColsPerTile           = ColsPerTile_;
    static constexpr int kGroupsPerStage        = GroupsPerStage_;
    static constexpr int kPipelineStages        = PipelineStages_;
    static constexpr Cache kCodeCache           = CodeCache_;
    static constexpr int kLaunchBoundsMinBlocks = LaunchBoundsMinBlocks_;

    static constexpr int kCtaWarps = kRowsPerCta;
    static constexpr int kThreads  = kCtaWarps * 32;
    static constexpr int kStageK   = kGroupsPerStage * Storage::kGroupK;
    static constexpr int kCodeVecsPerStage =
        kGroupsPerStage * Storage::kCodeBytesPerGroup / static_cast<int>(sizeof(uint4));
    static constexpr int kScalePairsPerStage = kGroupsPerStage / 2;
    static constexpr int kCodePhases         = (kGroupsPerStage + 3) / 4;
    // One lane owns one 32-bit word per phase, and a word is kCodesPerWord weights.
    static constexpr int kElementsPerLane = Storage::kCodesPerWord;
    static constexpr int kSharedBytes =
        kRowsPerCta * kPipelineStages *
        (kGroupsPerStage * Storage::kCodeBytesPerGroup +
         kScalePairsPerStage * static_cast<int>(sizeof(std::uint32_t)));

    static_assert(kRowsPerCta > 0 && kRowsPerCta <= 32);
    static_assert(kColsPerTile > 0 && kColsPerTile <= 8);
    static_assert(kGroupsPerStage > 0 && kGroupsPerStage % 2 == 0,
                  "ternary SIMT stages load aligned pairs of FP16 scales");
    static_assert(kPipelineStages >= 2 && kPipelineStages <= 8,
                  "ternary SIMT cp.async pipeline depth must fit cp_wait");
    static_assert(kLaunchBoundsMinBlocks >= 1);
    static_assert(kThreads <= 1024);
    static_assert(Storage::kCodeBytesPerGroup * Storage::kCodesPerByte == Storage::kGroupK,
                  "a code group must cover exactly one group of weights");
    static_assert(kCodeVecsPerStage * static_cast<int>(sizeof(uint4)) ==
                      kGroupsPerStage * Storage::kCodeBytesPerGroup,
                  "ternary code groups must decompose into complete 16-byte vectors");
    static_assert(32 * kElementsPerLane == 4 * Storage::kGroupK,
                  "one phase must cover four whole groups across the warp");
    static_assert(kCodePhases * 4 >= kGroupsPerStage,
                  "the phase walk must cover every staged group");
    static_assert(kSharedBytes <= 48 * 1024,
                  "ternary SIMT staged shared memory exceeds the static 48 KiB budget");
};

template <class Schedule>
__device__ __forceinline__ void ternary_simt_copy_code(uint4* shared_dst,
                                                       const std::uint8_t* global_src) {
    if constexpr (Schedule::kCodeCache == Cache::cg) {
        cp_async<16, Cache::cg>(shared_dst, global_src);
    } else {
        cp_async<16, Cache::ca>(shared_dst, global_src);
    }
}

template <class Schedule, bool FullStage>
__device__ __forceinline__ void ternary_simt_issue_stage(uint4* __restrict__ shared_codes,
                                                         std::uint32_t* __restrict__ shared_scales,
                                                         const std::uint8_t* __restrict__ code_row,
                                                         const std::uint8_t* __restrict__ scale_row,
                                                         int stage, int active_groups, int lane) {
    using Storage = typename Schedule::Storage;
    constexpr int kCodeVecs   = Schedule::kCodeVecsPerStage;
    constexpr int kScalePairs = Schedule::kScalePairsPerStage;

    const std::int64_t group0        = static_cast<std::int64_t>(stage) * Schedule::kGroupsPerStage;
    const std::uint8_t* stage_codes  = code_row + group0 * Storage::kCodeBytesPerGroup;
    const std::uint8_t* stage_scales = scale_row + group0 * Storage::kScaleBytesPerGroup;

    const int active_code_vecs = FullStage ? kCodeVecs
                                           : active_groups * Storage::kCodeBytesPerGroup /
                                                 static_cast<int>(sizeof(uint4));
    for (int vec = lane; vec < kCodeVecs; vec += 32) {
        if (FullStage || vec < active_code_vecs) {
            ternary_simt_copy_code<Schedule>(&shared_codes[vec],
                                             stage_codes + static_cast<std::int64_t>(vec) * 16);
        } else {
            shared_codes[vec] = uint4{0u, 0u, 0u, 0u};
        }
    }

    // Only whole, aligned pairs of binary16 scales are ever copied. active_groups is always even on
    // this path (kGroupsPerStage is even, and the launcher only selects a schedule whose
    // kGroupsPerStage divides the row's group count), so the trailing pair never straddles the row's
    // scale plane. Reading one group past the end to complete a pair would read past the tensor on
    // the final row, which is why the odd case is excluded rather than papered over.
    const int active_scale_pairs = FullStage ? kScalePairs : active_groups / 2;
    for (int pair = lane; pair < kScalePairs; pair += 32) {
        if (FullStage || pair < active_scale_pairs) {
            cp_async<4>(&shared_scales[pair], stage_scales + static_cast<std::int64_t>(pair) * 4);
        } else {
            shared_scales[pair] = 0u;
        }
    }
    cp_commit();
}

// `x_origin` already points at this stage's first K element of column 0 of the tile, and
// `x_col_stride` is the element distance between consecutive columns. That indirection lets the
// same body serve either a direct global-memory read (origin = x + col0*k + stage*kStageK,
// stride = k) or a block-staged shared-memory read (origin = the shared tile, stride = kStageK)
// without duplicating the dequant/FMA math.
template <class Schedule, bool FullStage, bool FullCols>
__device__ __forceinline__ void
ternary_simt_consume_stage(const __nv_bfloat16* __restrict__ x_origin, std::int64_t x_col_stride,
                           int active_cols, int active_groups,
                           const uint4* __restrict__ shared_codes,
                           const std::uint32_t* __restrict__ shared_scales, int lane,
                           float (&acc)[Schedule::kColsPerTile]) {
    using Storage = typename Schedule::Storage;
    constexpr int kCols       = Schedule::kColsPerTile;
    constexpr int kCodePhases = Schedule::kCodePhases;
    constexpr int kLaneK      = Schedule::kElementsPerLane;

#pragma unroll
    for (int phase = 0; phase < kCodePhases; ++phase) {
        const int group        = phase * 4 + (lane >> 3);
        const int stage_groups = FullStage ? Schedule::kGroupsPerStage : active_groups;
        if (group < stage_groups) {
            const std::uint32_t packed =
                reinterpret_cast<const std::uint32_t*>(shared_codes)[phase * 32 + lane];
            const std::uint32_t scale_pair = shared_scales[group >> 1];
            const std::uint16_t scale_bits =
                static_cast<std::uint16_t>(scale_pair >> ((group & 1) * 16));

            float weights[Storage::kCodesPerWord];
            Schedule::Atom::decode_sixteen(packed, scale_bits, weights);

            // Word `lane` of the phase is word (lane & 7) of group (phase * 4 + (lane >> 3)):
            // sixteen consecutive codes starting at k = group * 128 + (lane & 7) * 16.
            const std::int64_t xk = static_cast<std::int64_t>(phase) * kLaneK * 32 +
                                    static_cast<std::int64_t>(lane) * kLaneK;
#pragma unroll
            for (int col = 0; col < kCols; ++col) {
                if (FullCols || col < active_cols) {
                    const __nv_bfloat16* column =
                        x_origin + static_cast<std::int64_t>(col) * x_col_stride + xk;
                    const uint4 lo = load_vec<uint4>(column);
                    const uint4 hi = load_vec<uint4>(column + 8);
                    const float2 a0 = bf16x2_bits_to_float2(lo.x);
                    const float2 a1 = bf16x2_bits_to_float2(lo.y);
                    const float2 a2 = bf16x2_bits_to_float2(lo.z);
                    const float2 a3 = bf16x2_bits_to_float2(lo.w);
                    const float2 a4 = bf16x2_bits_to_float2(hi.x);
                    const float2 a5 = bf16x2_bits_to_float2(hi.y);
                    const float2 a6 = bf16x2_bits_to_float2(hi.z);
                    const float2 a7 = bf16x2_bits_to_float2(hi.w);
                    acc[col] = fmaf(weights[0], a0.x, acc[col]);
                    acc[col] = fmaf(weights[1], a0.y, acc[col]);
                    acc[col] = fmaf(weights[2], a1.x, acc[col]);
                    acc[col] = fmaf(weights[3], a1.y, acc[col]);
                    acc[col] = fmaf(weights[4], a2.x, acc[col]);
                    acc[col] = fmaf(weights[5], a2.y, acc[col]);
                    acc[col] = fmaf(weights[6], a3.x, acc[col]);
                    acc[col] = fmaf(weights[7], a3.y, acc[col]);
                    acc[col] = fmaf(weights[8], a4.x, acc[col]);
                    acc[col] = fmaf(weights[9], a4.y, acc[col]);
                    acc[col] = fmaf(weights[10], a5.x, acc[col]);
                    acc[col] = fmaf(weights[11], a5.y, acc[col]);
                    acc[col] = fmaf(weights[12], a6.x, acc[col]);
                    acc[col] = fmaf(weights[13], a6.y, acc[col]);
                    acc[col] = fmaf(weights[14], a7.x, acc[col]);
                    acc[col] = fmaf(weights[15], a7.y, acc[col]);
                }
            }
        }
    }
}

template <class Schedule, bool Full>
__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::
        kLaunchBoundsMinBlocks) void ternary_rowsplit_gemm_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                                                       const std::uint8_t* __restrict__ codes,
                                                                       const std::uint8_t* __restrict__ scales,
                                                                       __nv_bfloat16* __restrict__ out,
                                                                       std::int32_t out_ld,
                                                                       std::int32_t rows,
                                                                       std::int32_t k,
                                                                       std::int32_t cols,
                                                                       std::int32_t padded_k) {
    using Storage = typename Schedule::Storage;
    constexpr bool kFull              = Full;
    constexpr int kRowsPerCta         = Schedule::kRowsPerCta;
    constexpr int kColsPerTile        = Schedule::kColsPerTile;
    constexpr int kGroupsPerStage     = Schedule::kGroupsPerStage;
    constexpr int kPipelineStages     = Schedule::kPipelineStages;
    constexpr int kPipelinePrefetch   = kPipelineStages - 1;
    constexpr int kCodeVecsPerStage   = Schedule::kCodeVecsPerStage;
    constexpr int kScalePairsPerStage = Schedule::kScalePairsPerStage;

    __shared__ __align__(16) uint4 shared_codes[kRowsPerCta][kPipelineStages][kCodeVecsPerStage];
    __shared__ __align__(16)
        std::uint32_t shared_scales[kRowsPerCta][kPipelineStages][kScalePairsPerStage];
#ifdef NINFER_VOLTA_BUILD
    // One stage's K-slice of activations for the whole column tile, shared by every warp in the
    // CTA. Every warp owns a different output row but reads the same activation slice, so reading x
    // straight from global costs kRowsPerCta-fold redundant L1/LSU traffic. bf16 is kept as the
    // staged type so the consume path's uint4 vector loads and its exact bf16->float conversion are
    // both bit-for-bit unchanged -- this is a pure traffic optimisation with no numerical effect.
    __shared__ __align__(16) __nv_bfloat16 x_stage[kColsPerTile * Schedule::kStageK];
    static_assert(kColsPerTile * Schedule::kStageK * static_cast<int>(sizeof(__nv_bfloat16)) +
                          Schedule::kSharedBytes <=
                      48 * 1024,
                  "ternary SIMT activation staging must fit the static shared budget");
#endif

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerCta + warp;
    // A warp whose row is past the end must NOT return early: the activation staging below is a
    // block-wide cooperative step with __syncthreads(), and an early return would deadlock the
    // barrier for the surviving warps. Instead the row is clamped to a valid one (so every load
    // stays in bounds) and only the epilogue store is suppressed.
    const bool row_active = kFull || row < rows;
    const int safe_row    = row_active ? row : (rows > 0 ? rows - 1 : 0);

    const int col0        = static_cast<int>(blockIdx.y) * kColsPerTile;
    const int active_cols = kFull ? kColsPerTile : min(kColsPerTile, cols - col0);

    const int padded_groups = padded_k / Storage::kGroupK;
    const int groups        = k / Storage::kGroupK;
    const int stages =
        kFull ? groups / kGroupsPerStage : (groups + kGroupsPerStage - 1) / kGroupsPerStage;

    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(safe_row) * padded_groups *
                                               Storage::kCodeBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(safe_row) * padded_groups *
                                                 Storage::kScaleBytesPerGroup;

    float acc[kColsPerTile];
#pragma unroll
    for (int col = 0; col < kColsPerTile; ++col) { acc[col] = 0.0f; }

#pragma unroll
    for (int prefetch = 0; prefetch < kPipelinePrefetch; ++prefetch) {
        if (prefetch < stages) {
            const int active_groups =
                kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - prefetch * kGroupsPerStage);
            ternary_simt_issue_stage<Schedule, kFull>(shared_codes[warp][prefetch],
                                                      shared_scales[warp][prefetch], code_row,
                                                      scale_row, prefetch, active_groups, lane);
        } else {
            cp_commit();
        }
    }

#pragma unroll 1
    for (int stage = 0; stage < stages; ++stage) {
        const int fetch = stage + kPipelinePrefetch;
        if (fetch < stages) {
            const int active_groups =
                kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - fetch * kGroupsPerStage);
            const int buffer = fetch % kPipelineStages;
            ternary_simt_issue_stage<Schedule, kFull>(shared_codes[warp][buffer],
                                                      shared_scales[warp][buffer], code_row,
                                                      scale_row, fetch, active_groups, lane);
        } else {
            cp_commit();
        }

        cp_wait<kPipelinePrefetch>();
        __syncwarp();

        const int active_groups =
            kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - stage * kGroupsPerStage);
        const int buffer = stage % kPipelineStages;
#ifdef NINFER_VOLTA_BUILD
        __syncthreads();
        {
            const int staged_vecs = (active_groups * Storage::kGroupK) / 8;
            const std::int64_t src_k0 = static_cast<std::int64_t>(stage) * Schedule::kStageK;
            for (int i = static_cast<int>(threadIdx.x); i < active_cols * staged_vecs;
                 i += static_cast<int>(blockDim.x)) {
                const int col = i / staged_vecs;
                const int j   = i - col * staged_vecs;
                const uint4 v = load_vec<uint4>(
                    x + static_cast<std::int64_t>(col0 + col) * k + src_k0 + j * 8);
                *reinterpret_cast<uint4*>(&x_stage[col * Schedule::kStageK + j * 8]) = v;
            }
        }
        __syncthreads();
        ternary_simt_consume_stage<Schedule, kFull, kFull>(
            x_stage, Schedule::kStageK, active_cols, active_groups, shared_codes[warp][buffer],
            shared_scales[warp][buffer], lane, acc);
        __syncthreads();
#else
        ternary_simt_consume_stage<Schedule, kFull, kFull>(
            x + static_cast<std::int64_t>(col0) * k +
                static_cast<std::int64_t>(stage) * Schedule::kStageK,
            k, active_cols, active_groups, shared_codes[warp][buffer], shared_scales[warp][buffer],
            lane, acc);
        __syncwarp();
#endif
    }

#pragma unroll
    for (int col = 0; col < kColsPerTile; ++col) {
        if (kFull || col < active_cols) {
            const float sum = warp_reduce_sum(acc[col]);
            if (lane == 0 && row_active) {
                out[static_cast<std::int64_t>(col0 + col) * out_ld + row] = __float2bfloat16(sum);
            }
        }
    }
}

} // namespace ninfer::ops::detail
