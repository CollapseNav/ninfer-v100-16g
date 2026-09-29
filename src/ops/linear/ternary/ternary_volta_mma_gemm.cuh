#pragma once

// Ternary PQ2_0_G128 x BF16 GEMM on Volta tensor cores (mma.sync.m8n8k4), sm_70 only.
//
// The tensor-core sibling of the SIMT routes this port shipped first. Every SIMT route in this
// lineage decodes each weight to FP32 and then does one FP32 FMA, which caps prefill at ~38% of the
// V100's FP32 peak (6.03 TFLOP/s at 111.7 tok/s, against llama.cpp's 661.8 tok/s on the same prompt,
// which is 2.3x *above* FP32 peak and therefore integer or tensor-core work). Two unrelated SIMT
// shapes both landed on 111-115 tok/s, so the wall was the arithmetic, not the schedule.
//
// STRUCTURE. Lifted from q4_volta_mma_gemm.cuh, which is the same kernel for Q4 on this same card.
// Only the code fetch and the decode change:
//
//   * PQ2 packs four 2-bit codes per byte, so a 32-weight kKStep is 8 code bytes per row and one
//     lane's share is a single aligned 16-bit load -- half of Q4's 4-byte share, because the
//     ternary group is twice as wide (128 vs 64) at the same 32 code bytes.
//   * The decode uses the same "0x6400|n is exactly 1024+n as fp16" identity. Codes are unsigned
//     0..3 already, so subtracting 1025.0 yields code-1 directly and one __hmul2 by the group scale
//     finishes it.
//
// THE TWO SHAPE KNOBS, and why they are the whole story.
//
// `volta_mma_qk` has a fixed 32x8 output tile: the A operand (activation) is 32 tokens tall and the
// B operand (weights) is 8 output rows. Per k-slice, volta_load_qp and volta_load_k each read 512 B
// per warp. With kWarps warps, kTSub 32-token A sub-tiles per CTA, and 4 k-slices per kStep:
//
//     A bytes  = kWarps * kTSub * 4 * 512        (activation: read once per warp per sub-tile)
//     B bytes  = kWarps *        4 * 512        (weights: read once per warp, reused by every sub-tile)
//     FLOP     = kWarps * 8 * (32*kTSub) * 32 * 2
//     FLOP/byte = 8 * kTSub / (kTSub + 1)       -- independent of kWarps
//
// So kTSub is the only term that improves the shared-traffic ratio, and it saturates: 4.0 -> 5.33 ->
// 6.4 -> 7.1 -> 8.0 for kTSub = 1, 2, 4, 8, inf. And kWarps does NOT change the ratio -- it only
// buys occupancy, because the shared footprint per CTA grows with kWarps while the work does too.
//
// That is exactly the trade this file has to get right, and it is measured rather than derived:
// kTSub=4 at kWarps=4 reaches 6.4 FLOP/byte but drops to 3 CTAs/SM (18.75% occupancy) because the
// staging costs 25.6 KiB. Raising kWarps restores occupancy at the same ratio.
//
// Measured on this card at a 3412-token prompt: kWarps=4 kTSub=1 -> 462 t/s, kTSub=2 -> 582,
// kTSub=4 -> 636. Real, but sub-linear against the FLOP/byte prediction (1.33/1.60 predicted versus
// 1.26/1.38 measured), which is the occupancy term biting. See the variant table in the .cu.
//
// Numerics. The ternary weight is exact in fp16: code-1 is in {-1, 0, +1} for the codes the artifact
// contains, and (+/-scale) and (2*scale) are exact fp16 values because the group scale is already
// fp16. So (code-1)*scale is an exact fp16 result, and the only deviation from the FP32 SIMT route
// is the accumulation order in the tensor core's fp32 adder. Measured: with --greedy every variant
// here reproduces the FP32 reference kernel's token ids byte for byte.
//
// Activation staging converts bf16 -> fp16. bf16 has 8 mantissa bits and fp16 has 11, so the
// conversion is exact for the exponent range fp16 covers; post-RMSNorm activations of order 1 are
// inside it. Same conversion, same reasoning, as the Q4 kernel.

#include "ops/common/volta_mma.cuh"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

// Instruction-fixed constants. kSubTile is the mma A operand's row count and cannot change;
// kKStep is 32 because a kStep must lie inside one 128-weight group and 32 is the largest power of
// two dividing 128 that keeps one lane's code share a single aligned 16-bit load.
inline constexpr int kTernaryMmaSubTile = 32;
inline constexpr int kTernaryMmaKStep   = 32;
inline constexpr int kTernaryMmaXPad    = 8; // shared row padding. Unpadded, every lane of the
                                             // A-fragment load hits bank 0 (32-way conflict).
                                             // Padded, 8 distinct banks remain -- the hardware floor,
                                             // not a residual conflict: 32 lanes x 16 B = 512 B
                                             // against 128 B/cycle of shared bandwidth, so 4 cycles
                                             // is already optimal.
inline constexpr int kVoltaSharedPerSm = 96 * 1024;
inline constexpr int kVoltaThreadsPerSm = 2048;

// One lane owns one aligned 16-bit code word per kStep: kKStep weights at four codes per byte is
// 8 code bytes per row, and 8 rows x 8 bytes = 64 bytes is exactly 32 lanes x 2 bytes.
inline constexpr int kTernaryMmaCodeBytesPerKStep =
    kTernaryMmaKStep / PQ2RowSplitStorage::kCodesPerByte;
static_assert(kTernaryMmaCodeBytesPerKStep == 8, "a 32-weight kStep must be 8 code bytes");
static_assert(kTernaryMmaKStep % PQ2RowSplitStorage::kCodesPerWord == 0,
              "a kStep must hold whole 32-bit code words");
static_assert(PQ2RowSplitStorage::kGroupK % kTernaryMmaKStep == 0,
              "a kStep must lie inside one quantisation group");

template <int kWarps, int kTSub>
struct TernaryVoltaMmaShape {
    static constexpr int kRowsPerCta = kWarps * 8;
    static constexpr int kThreads    = kWarps * 32;
    static constexpr int kTTile      = kTernaryMmaSubTile * kTSub;
    static constexpr int kVecs       = kTernaryMmaKStep / 8;
    // Activation staging total vectors is kTTile * kVecs; each thread carries a fixed share.
    static constexpr int kVecsPerThread = kTTile * kVecs / kThreads;
    static constexpr int kSharedBytes =
        2 * kTTile * (kTernaryMmaKStep + kTernaryMmaXPad) * static_cast<int>(sizeof(__half)) +
        2 * kWarps * 8 * (kTernaryMmaKStep + kTernaryMmaXPad) * static_cast<int>(sizeof(__half));

    static_assert(kWarps > 0 && kWarps <= 16);
    static_assert(kTSub > 0 && kTSub <= 8);
    static_assert(kTTile * kVecs % kThreads == 0,
                  "activation staging must divide evenly across the CTA");
    static_assert(kVecsPerThread >= 1 && kVecsPerThread <= 8,
                  "the global-load carrier must stay small enough not to cost occupancy");

    // The occupancy ceiling shared memory imposes. It is NOT used directly as __launch_bounds__'s
    // min-blocks: register pressure binds before shared here, and forcing the theoretical maximum
    // would make the compiler cap registers and spill. The .cu picks the min-blocks per variant.
    static constexpr int kCtaByShared  = kVoltaSharedPerSm / kSharedBytes;
    static constexpr int kCtaByThreads = kVoltaThreadsPerSm / kThreads;
};

// kProbe is a TIMING PROBE selector, not a code path, and its outputs are numerically wrong by
// design. It exists because the fragment accounting says shared loads and the tensor core run at
// 1:1 at kTSub=4 (4 cycles each per k-slice), which caps this family near 50%, while the kernel
// measures 28% -- so the missing 22 points have to be located before they can be optimized.
//   1  kNoDecode -- store zeros instead of decoding the code word, which deletes the whole
//                   magic-number decode from the staging path
//   2  kNoALoad  -- load the A fragment once for the entire k walk instead of per k-slice
enum TernaryMmaProbe { kProbeOff = 0, kProbeNoDecode = 1, kProbeNoALoad = 2 };

// kCommit selects where the weight decode and its shared store sit relative to the MMA loop:
//   0 kCommitTrailing -- decode and store after the MMA loop (the original; fully exposed)
//   1 kCommitEarly    -- decode and store right after the prefetch, so ptxas can interleave the
//                        decode arithmetic with the MMA that only reads the OTHER buffer
//   2 kCommitSplit    -- decode into registers before the MMA loop and store only after it, so the
//                        exposed tail is one shared store instead of a decode plus a store
enum TernaryMmaCommit { kCommitTrailing = 0, kCommitEarly = 1, kCommitSplit = 2 };

template <class Shape, int kMinBlocks, bool kDirect, int kProbe = kProbeOff, int kCommit = kCommitTrailing>
__global__ __launch_bounds__(Shape::kThreads, kMinBlocks)
void ternary_volta_mma_gemm_kernel(const std::uint8_t* __restrict__ codes,
                                   const std::uint8_t* __restrict__ scales,
                                   const __nv_bfloat16* __restrict__ x,
                                   float* __restrict__ partial, __nv_bfloat16* __restrict__ out,
                                   int out_ld, int n, int k, int t, int padded_groups,
                                   int splits) {
    constexpr int kWarps  = Shape::kThreads / 32;
    constexpr int kTSub   = Shape::kTTile / kTernaryMmaSubTile;
    constexpr int kTTile  = Shape::kTTile;
    constexpr int kKStep  = kTernaryMmaKStep;
    constexpr int kXPad   = kTernaryMmaXPad;
    constexpr int kVecs   = Shape::kVecs;
    constexpr int kGroupK = PQ2RowSplitStorage::kGroupK;
    constexpr int kCodeB  = PQ2RowSplitStorage::kCodeBytesPerGroup;
    constexpr int kScaleB = PQ2RowSplitStorage::kScaleBytesPerGroup;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int n0   = (static_cast<int>(blockIdx.x) * kWarps + warp) * 8;
    const int t0   = static_cast<int>(blockIdx.z) * kTTile;
    const int tcnt = min(kTTile, t - t0);

    __shared__ __align__(16) __half x_sh[2][kTTile][kKStep + kXPad];
    __shared__ __align__(16) __half w_sh[2][kWarps][8][kKStep + kXPad];

    const int chunk  = (k / splits) & ~(kKStep - 1);
    const int kstart = static_cast<int>(blockIdx.y) * chunk;
    const int kend   = (static_cast<int>(blockIdx.y) == splits - 1) ? k : kstart + chunk;
    if (kstart >= kend) { return; }

    // Global-load half and decode-and-store half are held apart by this register carrier, for the
    // reason documented at length in q4_volta_mma_gemm.cuh: fusing them put 6.58 of 15.8
    // warp-cycles-per-issued-instruction in long_scoreboard, because the shared store waited on the
    // global load with no independent work in between.
    struct Carry {
        uint4 xraw[Shape::kVecsPerThread];
        std::uint16_t p2;
        __half sc;
    };
    Carry carry;

    auto prefetch = [&](int kbase, Carry& r) {
#pragma unroll
        for (int v = 0; v < Shape::kVecsPerThread; ++v) {
            // Flattened vector index across the whole CTA tile: row = i / kVecs, column = i % kVecs.
            const int i   = static_cast<int>(threadIdx.x) + v * Shape::kThreads;
            const int row = i / kVecs;
            if (row < tcnt) {
                r.xraw[v] = *reinterpret_cast<const uint4*>(
                    x + static_cast<std::int64_t>(t0 + row) * k + kbase + (i % kVecs) * 8);
            }
        }
        // 8 rows x kKStep weights, fetched once for the warp and reused across every sub-tile.
        // volta_load_k's I-major-mirrored B fragment gives only 8 distinct rows across the 32 lanes,
        // so doing this per lane would repeat the work 4x.
        //
        // `boff` is the byte offset of the kStep's first weight inside its group: four codes per
        // byte, so weight (kbase % kGroupK) lives in byte (kbase % kGroupK) / 4, and lane (lane & 3)
        // adds its own 2-byte share. Lane (row, c) therefore covers weights [8c, 8c+8) of the
        // kStep, which is exactly where commit() stores them.
        const int g    = kbase / kGroupK;
        const int boff = (kbase % kGroupK) >> 2;
        const int row  = n0 + (lane >> 2);
        r.p2           = 0;
        r.sc           = __ushort_as_half(0);
        if (row < n) {
            const std::uint8_t* crow =
                codes + static_cast<std::int64_t>(row) * padded_groups * kCodeB;
            r.p2 = *reinterpret_cast<const std::uint16_t*>(
                crow + static_cast<std::int64_t>(g) * kCodeB + boff + (lane & 3) * 2);
            r.sc = __ushort_as_half(
                reinterpret_cast<const std::uint16_t*>(scales + static_cast<std::int64_t>(row) *
                                                                    padded_groups * kScaleB)[g]);
        }
    };

    // Magic-number decode kept entirely in fp16, the ternary twin of Q4's. 0x6400|n is exactly
    // 1024+n as fp16 for n in [0,4), and 1025.0 is exact, so one __hsub2 against it yields code-1
    // with no rounding (both operands and the difference are small exact integers). One __hmul2 then
    // applies the group scale, and (+/-scale) and (2*scale) are exact fp16 values, so the product is
    // the correctly-rounded fp16 result and matches the FP32 route's operand bit for bit.
    // Weight decode into registers only. Split out from the store so the two halves can be placed
    // on opposite sides of the MMA loop.
    auto decode_weights = [&](const Carry& r, half2 (&decoded)[4]) {
        const std::uint32_t p2 = r.p2;
        const half2 sc2        = __half2half2(r.sc);
        const half2 bias       = __half2half2(__ushort_as_half(0x6401)); // 1025.0
        if constexpr (kProbe == kProbeNoDecode) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { decoded[j] = __half2half2(__ushort_as_half(0)); }
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                // Codes 2j and 2j+1 of the word, i.e. weights (lane&3)*8 + 2j and +1.
                const std::uint32_t pair = (p2 >> (4 * j)) & 0x0fu;
                const std::uint32_t lo   = pair & 0x3u;
                const std::uint32_t hi   = (pair >> 2) & 0x3u;
                const std::uint32_t bits = (0x6400u | lo) | ((0x6400u | hi) << 16);
                decoded[j] = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&bits), bias), sc2);
            }
        }
    };

    // Activation staging into shared, plus the weight store. Separate lambdas so kCommitSplit can
    // move only the weight store.
    auto store_weights = [&](const half2 (&decoded)[4], int buf) {
        *reinterpret_cast<uint4*>(&w_sh[buf][warp][lane >> 2][(lane & 3) * 8]) =
            *reinterpret_cast<const uint4*>(decoded);
    };

    auto store_activations = [&](const Carry& r, int buf) {
#pragma unroll
        for (int v = 0; v < Shape::kVecsPerThread; ++v) {
            const int i   = static_cast<int>(threadIdx.x) + v * Shape::kThreads;
            const int row = i / kVecs;
            if (row < tcnt) {
                const auto* src = reinterpret_cast<const __nv_bfloat16*>(&r.xraw[v]);
                __half tmp[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) { tmp[j] = __float2half(__bfloat162float(src[j])); }
                *reinterpret_cast<uint4*>(&x_sh[buf][row][(i % kVecs) * 8]) =
                    *reinterpret_cast<const uint4*>(tmp);
            }
        }
    };

    auto commit = [&](const Carry& r, int buf) {
        store_activations(r, buf);
        half2 decoded[4];
        decode_weights(r, decoded);
        store_weights(decoded, buf);
    };

    float d[kTSub][8];
#pragma unroll
    for (int sub = 0; sub < kTSub; ++sub) {
#pragma unroll
        for (int l = 0; l < 8; ++l) { d[sub][l] = 0.0f; }
    }

    prefetch(kstart, carry);
    commit(carry, 0);
    int buf = 0;
    // One barrier per iteration is enough: iteration i reads buf and writes buf^1, and the only
    // conflict -- iteration i-1's reads of the buffer iteration i is about to write -- is separated
    // by the barrier at the top, because that whole body precedes it.
    for (int kbase = kstart; kbase < kend; kbase += kKStep) {
        __syncthreads();
        const int nxt       = kbase + kKStep;
        const bool has_next = nxt < kend;
        if (has_next) { prefetch(nxt, carry); }

        half2 decoded_carry[4];
        if (has_next && kCommit == kCommitEarly) {
            store_activations(carry, buf ^ 1);
            decode_weights(carry, decoded_carry);
            store_weights(decoded_carry, buf ^ 1);
        } else if (has_next && kCommit == kCommitSplit) {
            store_activations(carry, buf ^ 1);
            decode_weights(carry, decoded_carry);
        }

#pragma unroll
        for (int kk = 0; kk < kKStep; kk += 8) {
            // B once per k-slice, then reused by every sub-tile. This hoist is the entire point of
            // kTSub, and B is the only term that amortises: A work grows with the token count.
            half2 b[4];
            volta_load_k(b, reinterpret_cast<const half2*>(&w_sh[buf][warp][0][kk]),
                         (kKStep + kXPad) / 2);
#pragma unroll
            for (int sub = 0; sub < kTSub; ++sub) {
                half2 a[4];
                if constexpr (kProbe == kProbeNoALoad) {
                    // PROBE: one A fragment for the whole k walk, so the per-k-slice shared A
                    // traffic disappears while the MMA count is unchanged.
                    if (kk == 0) {
                        volta_load_qp(
                            a, reinterpret_cast<const half2*>(&x_sh[buf][sub * kTernaryMmaSubTile][0]),
                            (kKStep + kXPad) / 2);
                    }
                } else {
                    volta_load_qp(
                        a, reinterpret_cast<const half2*>(&x_sh[buf][sub * kTernaryMmaSubTile][kk]),
                        (kKStep + kXPad) / 2);
                }
                volta_mma_qk(d[sub], a, b);
            }
        }

        if (has_next && kCommit == kCommitTrailing) {
            commit(carry, buf ^ 1);
        } else if (has_next && kCommit == kCommitSplit) {
            store_weights(decoded_carry, buf ^ 1);
        }
        buf ^= 1;
    }

#pragma unroll
    for (int sub = 0; sub < kTSub; ++sub) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int row_t = sub * kTernaryMmaSubTile + volta_d_get_i(l);
            const int col_n = n0 + volta_d_get_j(l);
            if (row_t < tcnt && col_n < n) {
                if constexpr (kDirect) {
                    out[static_cast<std::int64_t>(t0 + row_t) * out_ld + col_n] =
                        __float2bfloat16(d[sub][l]);
                } else {
                    atomicAdd(&partial[static_cast<std::int64_t>(t0 + row_t) * n + col_n],
                              d[sub][l]);
                }
            }
        }
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
