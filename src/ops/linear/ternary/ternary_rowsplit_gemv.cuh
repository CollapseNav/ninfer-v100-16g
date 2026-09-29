// Ported to the Volta (sm_70) tree from the Ambolio/ninfer-4090-windows ternary lineage.
// The original banner read "Ada sm_89"; this copy is built and verified for sm_70 only.
// See docs/ternary-port.md for what was changed and what was deliberately left out.
#pragma once

// Decode-shaped ternary GEMV (T == 1) for PQ2_0: one warp owns one output row.
//
// Why: the reference kernel in ternary_rowsplit_gemm.cuh gives each output row a full 128-thread
// CTA and seven block-wide barriers. That is fine for correctness but leaves decode far from the
// card's measured bandwidth. Here each lane covers four consecutive weights of every 128-group, so
// a warp reads exactly one 32-byte code span plus one 2-byte scale per group and reduces through
// shuffles with no __syncthreads at all.
//
// PQ2_0 packing makes the mapping exact rather than approximate: a group is 32 bytes holding four
// two-bit codes each, so for lane l the four columns 4l..4l+3 live entirely in byte l. Lane l
// loads one byte, decodes four weights, and consumes four bf16 activations.
//
// Occupancy, not instruction count, is what this kernel is tuned for: a GEMV needs many warps in
// flight to cover DRAM latency. A two-rows-per-warp variant (which reuses the activation loads)
// measured SLOWER (32.8 vs 46.4 t/s) because the extra accumulators and row bases cost registers
// and dropped the resident warp count. So: one row per warp, and only the micro-optimisations that
// remove instructions without adding state -- one 16-bit scale load, paired bf16 activation
// loads, and the per-group scale multiply hoisted out of the four FMAs.
//
// LAYOUT: ninfer/ggml put ne[0] on the contiguous axis, so a [k, 1] activation keeps element
// (column, 0) at column, and a one-token output row is simply out[row].

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Warps per block for the GEMV below.
inline constexpr int kGemvWarpsPerBlock = 8;

inline constexpr int kGemvCodeBytesPerGroup  = 32;
inline constexpr int kGemvScaleBytesPerGroup = 2;
inline constexpr int kGemvGroupK             = 128;

// Resident-CTA floor for the blocked prefill GEMV. Raising it forces ptxas to spend fewer
// registers, which buys resident warps at the cost of some recomputation. Measured on the target
// card: unpinned 116 regs -> 162.4 t/s, 3 CTAs 80 regs -> 175.4 t/s, 4 CTAs 64 regs -> 68.2 t/s.
// The last one spills 268 bytes per thread and loses more than the occupancy gains, so 3 is the
// measured optimum, not a conservative default.
inline constexpr int kBlockMinCtasPerSm = 3;

__device__ __forceinline__ float gemv_scale(const std::uint8_t* scale_ptr) {
    // One 16-bit load instead of two byte loads plus a shift/or; the scale plane is 2-byte aligned.
    const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(scale_ptr);
    return __half2float(__ushort_as_half(bits));
}

__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                             const std::uint8_t* __restrict__ codes,
                             const std::uint8_t* __restrict__ scales,
                             __nv_bfloat16* __restrict__ out, std::int32_t rows,
                             std::int32_t groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator = 0.0f;

    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        const float2 low        = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base));
        const float2 high = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base + 2));

        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const float dot = fmaf(static_cast<float>(static_cast<int>(raw & 3u) - 1), low.x,
                               fmaf(static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1),
                                    low.y,
                                    fmaf(static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1),
                                         high.x,
                                         static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1) *
                                             high.y)));
        accumulator = fmaf(gemv_scale(scale_row + group * kGemvScaleBytesPerGroup), dot,
                           accumulator);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) { out[warp] = __float2bfloat16_rn(accumulator); }
}

// Same T=1 GEMV, but with warps-per-block as a compile-time parameter so one build can sweep it.
//
// kGemvWarpsPerBlock=8 above came from the author's Ada card. The right value here is a measurement
// rather than an inheritance: this kernel is issue/occupancy-bound rather than bandwidth-bound (34
// bytes of codes plus 2 bytes of scales per 128 weights, and the activation row is re-read by every
// warp but is small enough to stay resident), so the number of resident warps is what decides
// throughput, and V100's shared/L1 split and scheduler count differ from Ada's.
// kSkipBias is a TIMING PROBE, not a code path. It drops the -1 that turns a stored code into a
// weight, which makes the output numerically wrong by design. It exists to answer exactly one
// question -- is this kernel's throughput set by instruction count? If removing four of its ~30
// inner-loop instructions does not speed it up, then cutting instructions further (dp4a, fp16x2)
// cannot pay, and the kernel is memory-latency bound instead. Never a default.
// kProbe is a TIMING PROBE selector, not a code path, and its output is numerically wrong by design.
// The standalone read-pattern microbenchmark (gemv_probe.cu) shows the code-row walk is already at
// 869.6 GB/s against a flat-stream ceiling of 882.6, so the loads are NOT the problem and widening
// them buys 3%. What is left is the other three loads and the arithmetic, so each is deleted in turn:
//   0  off
//   1  kNoActivation -- activation reads replaced by a constant (removes two L1 loads per group)
//   2  kNoCode       -- code byte replaced by a constant (removes the one DRAM load per group)
//   3  kNoScale      -- group scale replaced by 1.0 (removes the scale load per group)
//   4  kCodeFromAlu  -- code byte computed from (group, lane) instead of loaded. This is the one
//                       that separates the code LOAD from the code DECODE: kNoCode removes both and
//                       measures +53%, and only this arm says which half that is.
enum TernaryGemvProbe {
    kGemvProbeOff = 0,
    kGemvNoActivation = 1,
    kGemvNoCode = 2,
    kGemvNoScale = 3,
    kGemvCodeFromAlu = 4
};

// kRows output rows per warp. The activation window (4 bf16 per lane per group) is identical for
// every output row, so the one-row-per-warp form loads and converts it once per row; holding kRows
// rows in one warp loads it once for all of them and gives each warp kRows independent code streams.
//
// MEASURED AND REFUTED. rows=1 -> 41.6 t/s (72 registers), rows=2 -> 24.0 (74 registers),
// rows=4 -> 16.2 (85 registers). Register pressure binds harder than the shared activation helps:
// the default shape already sits at 72 registers, i.e. 28 of the 64 warps per SM, and asking each
// warp to do more work costs more resident warps than the saved instructions buy back. Kept as a
// selectable arm (NINFER_TERNARY_GEMV_ROWS) so the measurement can be repeated, never a default.
//
// kMinBlocks is `__launch_bounds__`'s second argument: ptxas then caps registers at
// 65536/(kThreads*kMinBlocks). At kThreads=256 that is 256/128/85/64/51/42/32 registers for
// kMinBlocks = 1/2/3/4/5/6/8, so only 4 and above constrain the shipped kernel's 72. kMinBlocks = 1
// is the previous single-argument form: the cap lands at 256, above the 255 architectural maximum,
// so it compiles exactly what it compiled before. This is the clean test of the memory-level-
// parallelism hypothesis: the unroll curve (1->29.9, 2->34.1, 4->40.4, 8->41.6, 12->33.9, 16->30.9,
// 20->25.7 t/s at 36/48/72/95/116/134 registers) says more in-flight loads per warp help until
// register pressure takes resident warps away, so forcing registers DOWN at unroll 8 buys warps
// without giving up the unroll. No semantic change either way.
template <int kWarps, bool kSkipBias = false, int kUnroll = 1, int kProbe = kGemvProbeOff,
          int kRows = 1, int kMinBlocks = 1>
__global__ __launch_bounds__(kWarps * 32, kMinBlocks)
void ternary_pq2_gemv_w_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out, std::int32_t rows,
                               std::int32_t groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5);
    const int row0 = warp * kRows;
    if (row0 >= rows) { return; }

    const std::uint8_t* code_row[kRows];
    const std::uint8_t* scale_row[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        if (row0 + r < rows) {
            code_row[r] = codes + static_cast<std::int64_t>(row0 + r) * groups_per_row *
                                      kGemvCodeBytesPerGroup;
            scale_row[r] = scales + static_cast<std::int64_t>(row0 + r) * groups_per_row *
                                        kGemvScaleBytesPerGroup;
        } else {
            code_row[r]  = codes; // never dereferenced: the row guard below skips the store
            scale_row[r] = scales;
        }
    }

    float accumulator[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) { accumulator[r] = 0.0f; }

    // kUnroll is the other lever on this kernel's memory-level parallelism. Two measurements rule out
    // resident warps (warps per block is flat to inverse); unrolling the group walk is what lets
    // ptxas keep several groups' loads in flight instead of one.
#pragma unroll(kUnroll)
    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        float2 low;
        float2 high;
        if constexpr (kProbe == kGemvNoActivation) {
            low  = make_float2(1.0F, 1.0F);
            high = make_float2(1.0F, 1.0F);
        } else {
            low  = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base));
            high = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base + 2));
        }
        const int bias = kSkipBias ? 0 : 1;

#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const bool live = (kProbe == kGemvNoActivation) || true;
            (void)live;
            if (row0 + r >= rows) { continue; }
            const std::uint8_t raw =
                kProbe == kGemvNoCode
                    ? static_cast<std::uint8_t>(0x55u)
                    : (kProbe == kGemvCodeFromAlu
                           ? static_cast<std::uint8_t>(group * 7 + lane)
                           : code_row[r][group * kGemvCodeBytesPerGroup + lane]);
            const float dot = fmaf(static_cast<float>(static_cast<int>(raw & 3u) - bias), low.x,
                                   fmaf(static_cast<float>(static_cast<int>((raw >> 2) & 3u) - bias),
                                        low.y,
                                        fmaf(static_cast<float>(
                                                 static_cast<int>((raw >> 4) & 3u) - bias),
                                             high.x,
                                             static_cast<float>(
                                                 static_cast<int>((raw >> 6) & 3u) - bias) *
                                                 high.y)));
            const float scale =
                kProbe == kGemvNoScale
                    ? 1.0F
                    : gemv_scale(scale_row[r] + group * kGemvScaleBytesPerGroup);
            accumulator[r] = fmaf(scale, dot, accumulator[r]);
        }
    }

#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        if (row0 + r >= rows) { continue; }
        float acc = accumulator[r];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        }
        if (lane == 0) { out[row0 + r] = __float2bfloat16_rn(acc); }
    }
}

// Small-token-tile variant, for the speculative VERIFY pass (T = draft + 1, i.e. 2..4).
//
// A per-token GEMV would re-read every weight for every token, and weights are exactly what the
// decode path is bound by, so a verify round would cost T full decode passes and speculation could
// never pay for itself. This kernel keeps the weights single-read: the code byte and scale are
// loaded and decoded once per group and reused for all kT tokens, while each token contributes its
// own four activations.
//
// It does NOT reach the weight-resident ideal, and the isolated measurement (tile_probe.cu) says why:
// the per-token cost is the activation load plus four conversions plus four FMAs, ~11 instructions
// against the ~28 the group itself costs, so M=2 lands at 1.67x the kT=1 tile rather than ~1.05x.
// What the probe did find is pure waste: instantiating kT=4 for a 2-token verify measured 1.283 ms
// against 1.024 ms for a real kT=2, so the launcher now picks kT from the token count.
// kUnroll is the group-walk unroll, and on this kernel it is the same lever the T=1 GEMV is tuned
// with -- that kernel's own curve is 1 -> 29.9, 2 -> 34.1, 4 -> 40.4, 8 -> 41.6, 12 -> 33.9, and the
// document calls it the only control on memory-level parallelism. The tile kernel carried NO pragma
// on this loop, so ptxas chose for itself while the per-token work (a load, two converts and five
// FMAs per token per group) was added on top. One extra token costs 0.60 of a whole T=1 step, which
// is what makes MTP lose on prose, so if that cost is a missing pipeline rather than real traffic
// this is where it shows.
// kT is capped at 16 because the context-lookup path verifies the full
// qwen3_6::kMtpLookupMaximumDrafts = 15 proposal window, i.e. T = 16. The row-blocked kernel was
// the only thing that could serve that width before, and it loses the whole T = 5..8 band to this
// kernel in situ, so the wide tile is the one to reach for.
// kShareActivation is a TIMING PROBE, not a code path: every token in the tile uses the first
// token's activation values, so the result is numerically wrong by construction. It exists to price
// the per-token side of this kernel, which the SASS says is the whole story: per (token, group) the
// loop body is ~5 FFMA of real work against ~5 PRMT for the bf16->fp32 converts, ~1.25 LDG.64,
// ~1.25 I2F, and ~11 instructions of address arithmetic and register moves. If removing the loads
// and the converts for all but one token does not move the needle, the fp32-activation idea (have
// the rotation emit fp32 so no GEMV has to convert) is not worth its plumbing.
template <int kT, int kUnroll = 4, bool kShareActivation = false>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_tile_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                  std::int32_t groups_per_row, std::int32_t tokens,
                                  std::int32_t out_row_stride) {
    static_assert(kT >= 1 && kT <= 16, "tile size must stay small enough to keep accumulators in registers");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;
    // One multiply per token, hoisted out of the group walk rather than recomputed inside it.
    const std::int64_t token_stride =
        static_cast<std::int64_t>(groups_per_row) * kGemvGroupK;
    const __nv_bfloat16* token_x[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { token_x[t] = x + static_cast<std::int64_t>(t) * token_stride; }

    float accumulator[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { accumulator[t] = 0.0f; }

#pragma unroll(kUnroll)
    for (int group = 0; group < groups_per_row; ++group) {
        // Weights: one byte of codes + one 16-bit scale, reused across every token in the tile.
        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const float scale = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
        const float weight0 = static_cast<float>(static_cast<int>(raw & 3u) - 1);
        const float weight1 = static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1);
        const float weight2 = static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1);
        const float weight3 = static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1);

        const std::int32_t base = group * kGemvGroupK + lane * 4;
        if constexpr (kShareActivation) {
            const uint2 packed = *reinterpret_cast<const uint2*>(token_x[0] + base);
            const float2 low =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed.x));
            const float2 high =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed.y));
#pragma unroll
            for (int t = 0; t < kT; ++t) {
                if (t < tokens) {
                    const float dot =
                        fmaf(weight0, low.x,
                             fmaf(weight1, low.y, fmaf(weight2, high.x, weight3 * high.y)));
                    accumulator[t] = fmaf(scale, dot, accumulator[t]);
                }
            }
        } else {
#pragma unroll
            for (int t = 0; t < kT; ++t) {
                if (t < tokens) {
                    // A lane's four activations are four contiguous bf16, so one 8-byte load replaces
                    // the two 4-byte ones. base is 8*lane + 256*group elements, i.e. always 8-byte
                    // aligned.
                    const uint2 packed = *reinterpret_cast<const uint2*>(token_x[t] + base);
                    const float2 low =
                        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed.x));
                    const float2 high =
                        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed.y));
                    const float dot =
                        fmaf(weight0, low.x,
                             fmaf(weight1, low.y, fmaf(weight2, high.x, weight3 * high.y)));
                    accumulator[t] = fmaf(scale, dot, accumulator[t]);
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kT; ++t) {
        float value = accumulator[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0 && t < tokens) {
            // Token-major output: element (row, token) lives at token * out_row_stride + row.
            out[static_cast<std::int64_t>(t) * out_row_stride + warp] =
                __float2bfloat16_rn(value);
        }
    }
}

// Token-blocked variant, for PREFILL: the prefill chunk.
//
// The verify kernel above takes the whole T at once and is capped at kT <= 8 by register
// pressure, so it cannot serve a prefill chunk -- the CLI requires that chunk to be a multiple
// of 128 (apps/cli/options.cpp: "--prefill-chunk must be a multiple of 128"). Without this
// kernel, every prefill therefore falls through to the correctness-first reference kernel in
// ternary_rowsplit_gemm.cuh, whose shape is one 128-thread CTA per output row plus seven
// block-wide barriers per row and a four-way duplicated code load (threads index>>2 share a
// byte). Published patch state measured that kernel at 42 GiB/s against this card's 638.7 GB/s
// sustained-read ceiling -- 7%.
//
// This kernel keeps the GEMV shape -- lane l reads code byte l and so covers columns 4l..4l+3,
// weights are decoded once per group and reused for the whole token tile, and the only reduction
// is five shuffle steps -- but tiles BOTH axes: the token axis across blockIdx.y (kT tokens) and
// the row axis across kR rows per warp.
//
// Why the row axis matters: measured on the target card the R=1/kT=8 shape reaches 125.7 t/s at
// an effective weight bandwidth of only 112 GB/s, while the T=1 GEMV reads at 422 GB/s. The
// kernel is therefore NOT weight-bandwidth-bound -- it is bound by the activation side, which
// loads kT*4 elements per lane per group against a single weight byte. Since activations are
// identical for every output row of the same token, widening the warp over rows amortises them:
//
//   loads per FMA  =  (2*kT + kR) / (4 * kR * kT)
//
// R=1,kT=8 gives 0.53; R=2,kT=8 gives 0.28; R=4,kT=4 gives 0.19. The cost is registers
// (kR*kT accumulators plus 2*kT activation pairs), which is why both axes are capped small.
//
// The min-CTAs bound matters more than it looks. ptxas gives the 4x8 shape 116 registers
// unpinned, which fits only two 256-thread CTAs per SM -- 16 of 48 warps. The T=1 GEMV below
// runs 43 registers at ~83% occupancy, and this file's own note says occupancy is what the
// GEMV family is tuned for. Pinning the bound makes ptxas trade registers for resident warps;
// the measured effect is in the plan document.
// kUnroll is the group-walk unroll. Unlike the small-tile kernel above, this one measures BEST
// UNROLLED AT DEPTH 1 -- i.e. with no pragma at all, which is what it shipped with. The token loop
// inside it is already fully unrolled at kT=8, so deepening the group walk multiplies an
// already-large body: on the repeated-sentence prompt at K=7 the arms measure depth 1 -> 62.9,
// 2 -> 46.9, 4 -> 34.9, 8 -> 34.7, 16 -> 18.8 t/s. The default is therefore 1 and
// NINFER_TERNARY_BLOCK_UNROLL exists only to keep the negative result reproducible.
template <int kR, int kT, int kUnroll = 1>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32, kBlockMinCtasPerSm)
void ternary_pq2_gemv_tile_block_kernel(const __nv_bfloat16* __restrict__ x,
                                        const std::uint8_t* __restrict__ codes,
                                        const std::uint8_t* __restrict__ scales,
                                        __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                        std::int32_t groups_per_row, std::int32_t tokens,
                                        std::int32_t out_row_stride) {
    static_assert(kR >= 1 && kR <= 8, "row block must keep accumulators in registers");
    static_assert(kT >= 1 && kT <= 8, "token block must keep accumulators in registers");
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int warp_in_block = static_cast<int>(threadIdx.x) >> 5;
    const int warp = static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + warp_in_block;
    const int row0    = warp * kR;
    const int token0  = static_cast<int>(blockIdx.y) * kT;
    // CTA-uniform early exit only: a per-warp exit here would leave the other warps waiting at the
    // flush barrier below. A partially covered CTA (rows not a multiple of kR*warps) keeps every
    // warp alive and guards its own work with `active` instead.
    if (static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock * kR >= rows) { return; }
    const bool active = row0 < rows;

    // Results land here first so the global write can be coalesced. Writing straight out from lane 0
    // costs one 2-byte store per (row, token) -- 32 of them per warp, each its own 32-byte sector,
    // which NCU flags as "only 2.0 of the 32 bytes transmitted per sector are utilized". A CTA owns
    // a contiguous row range and a contiguous token range, so staging and flushing per token turns
    // that into one 64-byte write per token.
    __shared__ __nv_bfloat16 staged[kT][kGemvWarpsPerBlock * kR];

    float accumulator[kR][kT];
#pragma unroll
    for (int r = 0; r < kR; ++r) {
#pragma unroll
        for (int t = 0; t < kT; ++t) { accumulator[r][t] = 0.0f; }
    }

#pragma unroll(kUnroll)
    for (int group = 0; active && group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        // Activations: loaded ONCE per group and reused by every row this warp owns. This is the
        // whole point of kR -- with kR == 1 they are re-loaded for each row the warp covers.
        float2 low[kT];
        float2 high[kT];
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            const std::int32_t token = token0 + t;
            low[t]  = make_float2(0.0f, 0.0f);
            high[t] = make_float2(0.0f, 0.0f);
            if (token < tokens) {
                const __nv_bfloat16* x_token =
                    x + static_cast<std::int64_t>(token) * groups_per_row * kGemvGroupK;
                low[t]  = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base));
                high[t] = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + 2));
            }
        }

#pragma unroll
        for (int r = 0; r < kR; ++r) {
            const int row = row0 + r;
            if (row < rows) {
                const std::uint8_t* code_row =
                    codes + static_cast<std::int64_t>(row) * groups_per_row *
                                kGemvCodeBytesPerGroup;
                const std::uint8_t* scale_row =
                    scales + static_cast<std::int64_t>(row) * groups_per_row *
                                 kGemvScaleBytesPerGroup;
                const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
                const float scale = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
                const float weight0 = static_cast<float>(static_cast<int>(raw & 3u) - 1);
                const float weight1 = static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1);
                const float weight2 = static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1);
                const float weight3 = static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1);
#pragma unroll
                for (int t = 0; t < kT; ++t) {
                    if (token0 + t < tokens) {
                        const float dot =
                            fmaf(weight0, low[t].x,
                                 fmaf(weight1, low[t].y,
                                      fmaf(weight2, high[t].x, weight3 * high[t].y)));
                        accumulator[r][t] = fmaf(scale, dot, accumulator[r][t]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < kR; ++r) {
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            float value = accumulator[r][t];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffffu, value, offset);
            }
            if (lane == 0 && active) {
                // Row order inside the CTA matches the global row order: warp_in_block owns rows
                // [warp_in_block*kR, ...) of the block's contiguous range.
                staged[t][warp_in_block * kR + r] = __float2bfloat16_rn(value);
            }
        }
    }
    __syncthreads();

    // Coalesced flush. Token-major output puts element (row, token) at token * out_row_stride + row,
    // and this CTA owns one contiguous row range, so a fixed token is a contiguous run.
    constexpr int kRowsPerBlock = kGemvWarpsPerBlock * kR;
    const int row_base          = static_cast<int>(blockIdx.x) * kRowsPerBlock;
#pragma unroll
    for (int i = threadIdx.x; i < kT * kRowsPerBlock; i += kGemvWarpsPerBlock * 32) {
        const int t     = i / kRowsPerBlock;
        const int r     = i % kRowsPerBlock;
        const int token = token0 + t;
        const int row   = row_base + r;
        if (token < tokens && row < rows) {
            out[static_cast<std::int64_t>(token) * out_row_stride + row] = staged[t][r];
        }
    }
}

} // namespace ninfer::ops::detail
