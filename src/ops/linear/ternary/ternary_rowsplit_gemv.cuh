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

// Epilogue store shared by the decode and verify GEMV kernels.
//
// kAddResidual = true folds ops::residual_add into the producer. The arithmetic is chosen, not
// inherited: ops::residual_add computes __float2bfloat16_rn(bf16(y) + bf16(x)) on a projection the
// GEMV has ALREADY rounded to bf16, so the composed route rounds twice. This epilogue reproduces
// exactly those two roundings -- round the fp32 accumulator to bf16 first, then add in fp32 and
// round again -- which makes the fused route bit-identical to the composed one. Adding the fp32
// accumulator to the residual instead would be strictly more accurate and is NOT what this does:
// it would move greedy ids and require a re-recorded md5 baseline and a perplexity check for a
// gain that the launch removal already delivers on its own.
//
// `residual` is deliberately not __restrict__: on the fused route it aliases `out`, and the load at
// `index` must be the value the composed route would have read from the projection scratch.
template <bool kAddResidual>
__device__ __forceinline__ void gemv_store(__nv_bfloat16* out, const __nv_bfloat16* residual,
                                           std::int64_t index, float value) {
    __nv_bfloat16 projected = __float2bfloat16_rn(value);
    if constexpr (kAddResidual) {
        projected = __float2bfloat16_rn(__bfloat162float(projected) +
                                        __bfloat162float(residual[index]));
    }
    out[index] = projected;
}

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

// table[byte] = the four (code-1) fp16 values of that code byte, packed as two half2 (low half =
// codes 0,1, high half = codes 2,3). 256 * 8 B = 2 KB, L1-resident, filled once by the launcher.
// Its only user is the tile kernel's kTableWeights arm; see the construction it replaces there.
extern __device__ uint2 kPq2TileWeights[256];

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
// kFpMode selects how the activation is consumed: 0 = bf16 container with the fp32 dot chain
// (today's default), 1 = fp16 container with the group's four-term dot done in half2, 2 = fp16
// container with the original fp32 chain. Modes 1 and 2 differ only in where the group sum is
// rounded: mode 1 rounds it once to fp16 and is therefore not bit-exact, mode 2 is bit-exact with
// mode 0 because both the container round trip and the half2 -> fp32 widening are exact.
// kTableWeights: 0 = build the four (code-1) fp16 values from the byte with the magic-number
// identity (the shipped path), 1 = read them from a 2 KB __device__ table (measured -28%, see the
// launcher), 2 = read them from a 2 KB SHARED-memory table built by the block once at entry. The
// decode construction is ~12.3 SASS instructions of a ~26.5-instruction group body, and the probes
// price it at ~23% of the T = 1 step, so replacing it with one LDS.64 is the largest single
// instruction-count lever left on the decode shape. Shared rather than global because the gather is
// 8 bytes per lane out of a 2 KB window: L1 tags cannot serve that (the == 1 arm), LDS can.
// kMinBlocks is __launch_bounds__'s second argument; the tile kernel had none, so ptxas picks 60
// registers for the fp16 decode shape (32 of 64 warps per SM). The dedicated T = 1 kernel gained
// +4.8% from exactly this cap, so it is worth a sweep here too.
template <int kT, int kUnroll = 4, bool kShareActivation = false, int kFpMode = 0,
          int kProbe = kGemvProbeOff, int kTableWeights = 0, int kMinBlocks = 1,
          bool kAddResidual = false>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32, kMinBlocks)
void ternary_pq2_gemv_tile_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                  std::int32_t groups_per_row, std::int32_t tokens,
                                  std::int32_t out_row_stride,
                                  const __nv_bfloat16* residual = nullptr) {
    static_assert(kT >= 1 && kT <= 16, "tile size must stay small enough to keep accumulators in registers");
    static_assert(!(kShareActivation && kFpMode != 0),
                  "the activation-sharing probe reads bf16 only; it is not valid with an fp16 tile");
    // Built before the early return below, which is per-warp: a __syncthreads() after it would hang
    // the warps that retire.
    __shared__ uint2 smem_weights[256];
    if constexpr (kTableWeights == 2) {
        if (static_cast<int>(threadIdx.x) < 256) {
            const std::uint32_t raw = static_cast<std::uint32_t>(threadIdx.x);
            const std::uint32_t lo  = 0x64006400u | (raw & 0x03u) | ((raw & 0x0Cu) << 14);
            const std::uint32_t hi  = 0x64006400u | ((raw >> 4) & 0x03u) | ((raw & 0xC0u) << 10);
            const __half2 w01 = __hsub2(*reinterpret_cast<const __half2*>(&lo),
                                        __float2half2_rn(1025.0F));
            const __half2 w23 = __hsub2(*reinterpret_cast<const __half2*>(&hi),
                                        __float2half2_rn(1025.0F));
            uint2 entry;
            entry.x = *reinterpret_cast<const std::uint32_t*>(&w01);
            entry.y = *reinterpret_cast<const std::uint32_t*>(&w23);
            smem_weights[threadIdx.x] = entry;
        }
        __syncthreads();
    }
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
        // PROBE arms, the same four the dedicated T=1 kernel carries (kProbe, compile-time, so
        // the shipped path is untouched when it is kGemvProbeOff). kGemvNoCode folds BOTH the byte
        // and its decode to constants -- 0xFF keeps the weights non-zero so the activation loads
        // stay live -- while kGemvCodeFromAlu keeps the decode and removes only the load; the
        // difference between those two arms is the decode arithmetic, which is the split the port
        // doc could never measure on the tile path because the arms never reached it.
        const std::uint8_t raw =
            kProbe == kGemvNoCode
                ? static_cast<std::uint8_t>(0xFFu)
                : (kProbe == kGemvCodeFromAlu
                       ? static_cast<std::uint8_t>(group * 7 + lane)
                       : code_row[group * kGemvCodeBytesPerGroup + lane]);
        const float scale = kProbe == kGemvNoScale
                                ? 1.0F
                                : gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
        const float weight0 = static_cast<float>(static_cast<int>(raw & 3u) - 1);
        const float weight1 = static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1);
        const float weight2 = static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1);
        const float weight3 = static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1);
        // (code - 1) as fp16 through the same magic number the prefill kernel uses: the bit pattern
        // 0x6400|n is exactly 1024+n as fp16, so one HADD2 against 1025.0 returns two exact
        // (code-1) values at once and the four I2F conversions disappear.
        __half2 w01;
        __half2 w23;
        if constexpr (kTableWeights == 2 && kFpMode != 0) {
            // One LDS.64 replaces the whole construction below.
            const uint2 from_smem = smem_weights[raw];
            w01 = *reinterpret_cast<const __half2*>(&from_smem.x);
            w23 = *reinterpret_cast<const __half2*>(&from_smem.y);
        } else if constexpr (kTableWeights == 1 && kFpMode != 0) {
            // WEIGHT TABLE arm: the same four (code-1) fp16 values in one LDG.64 instead of the
            // magic-number construction below. SASS puts that construction at ~12.3 instructions
            // per group inside a loop of ~26.5, and the nocode/codealu probe arms price the decode
            // at up to 24% of the whole decode step. Bit-identical: 0x6400|n minus 1025 in fp16 is
            // exactly n-1, and the table holds the fp16 bits of those same four values.
            const uint2 table_bits = kPq2TileWeights[raw];
            w01                    = *reinterpret_cast<const __half2*>(&table_bits.x);
            w23                    = *reinterpret_cast<const __half2*>(&table_bits.y);
        } else if constexpr (kFpMode != 0) {
            const std::uint32_t lo_bits = 0x64006400u | static_cast<std::uint32_t>(raw & 0x03u) |
                                          (static_cast<std::uint32_t>(raw & 0x0Cu) << 14);
            const std::uint32_t hi_bits =
                0x64006400u | static_cast<std::uint32_t>((raw >> 4) & 0x03u) |
                (static_cast<std::uint32_t>(raw & 0xC0u) << 10);
            w01 = __hsub2(*reinterpret_cast<const __half2*>(&lo_bits), __float2half2_rn(1025.0F));
            w23 = __hsub2(*reinterpret_cast<const __half2*>(&hi_bits), __float2half2_rn(1025.0F));
        }

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
                    float dot;
                    // PROBE (kGemvNoActivation): constant activation VALUES, no LDG and no
                    // convert, with the group walk, the code side, the scale and the dot chain
                    // still live -- so this arm prices the activation side only.
                    const uint2 packed =
                        kProbe == kGemvNoActivation
                            ? make_uint2(0x3C003C00u, 0x3C003C00u)
                            : *reinterpret_cast<const uint2*>(token_x[t] + base);
                    if constexpr (kFpMode == 1) {
                        // One half2 FMA pair per token: (w0*a0 + w2*a2, w1*a1 + w3*a3) in fp16, then
                        // the two lanes join in fp32, so the only rounding fp16 introduces is that
                        // one add -- the products are exact (a weight is -1, 0 or +1).
                        const __half2 a01 = *reinterpret_cast<const __half2*>(&packed.x);
                        const __half2 a23 = *reinterpret_cast<const __half2*>(&packed.y);
                        const float2 p    = __half22float2(__hfma2(w01, a01, __hmul2(w23, a23)));
                        dot               = p.x + p.y;
                    } else if constexpr (kFpMode == 2) {
                        // Same fp16 container, original fp32 chain: bit-exact with mode 0.
                        const float2 wf01 = __half22float2(w01);
                        const float2 wf23 = __half22float2(w23);
                        const float2 a01  = __half22float2(*reinterpret_cast<const __half2*>(&packed.x));
                        const float2 a23  = __half22float2(*reinterpret_cast<const __half2*>(&packed.y));
                        dot = fmaf(wf01.x, a01.x,
                                   fmaf(wf01.y, a01.y, fmaf(wf23.x, a23.x, wf23.y * a23.y)));
                    } else {
                        const float2 low =
                            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed.x));
                        const float2 high =
                            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed.y));
                        dot = fmaf(weight0, low.x,
                                   fmaf(weight1, low.y, fmaf(weight2, high.x, weight3 * high.y)));
                    }
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
            gemv_store<kAddResidual>(out, residual,
                                     static_cast<std::int64_t>(t) * out_row_stride + warp, value);
        }
    }
}

// Wide-lane T = 1 decode GEMV: two code bytes and sixteen activation bytes per lane per iteration,
// covering TWO groups. It exists because the shipped T = 1 path (the tile kernel at kT = 1, depth 8)
// is latency-bound rather than issue-bound, and the two levers that address issue count both failed:
// a shared-memory decode table cuts the per-group body from 29 SASS instructions to 22 and measured
// -23%, and every register cap that buys resident warps measured flat or worse. What this variant
// changes is BYTES PER LOAD INSTRUCTION -- the code read becomes one LDG.16 per lane (64 B per warp
// per instruction instead of 32) and the activation one LDG.128 (512 B instead of 256) -- so the
// same number of loads in flight carries twice the weight bytes, at ~25% fewer instructions per
// byte because the non-decode part of the body amortises over two groups.
//
// The mapping stays uniform. Lane l covers k in [8l, 8l+8) of the 256-k pair window for every l:
// for l < 16 that is offset 8l inside group 2p, for l >= 16 offset 8(l-16) inside group 2p+1, i.e.
// 8l in both cases, so the activation address is pair*256 + 8l and the code address pair*64 + 2l --
// the same expression for every lane, and lane l's scale is the one of group 2p + (l >> 4).
//
// Numerics: each four-weight dot is still built as one HMUL2/HFMA2 pair and summed in fp32, the
// same shape as the tile kernel's mode 1; only the association of the two four-dots inside one lane
// differs, so this is the same class of last-bit difference the fp16 container already carries.
// groups_per_row must be even (40/48/80/136 in this model); the launcher falls back otherwise.
template <int kUnroll = 4, int kMinBlocks = 1>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32, kMinBlocks)
void ternary_pq2_gemv_wide1_kernel(const __nv_bfloat16* __restrict__ x,
                                   const std::uint8_t* __restrict__ codes,
                                   const std::uint8_t* __restrict__ scales,
                                   __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                   std::int32_t groups_per_row, std::int32_t out_row_stride) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;
    const __nv_bfloat16* x_lane = x + lane * 8;

    const int pairs = groups_per_row >> 1;
    float accumulator = 0.0F;
#pragma unroll(kUnroll)
    for (int pair = 0; pair < pairs; ++pair) {
        const uint4 act = *reinterpret_cast<const uint4*>(x_lane + pair * 256);
        const std::uint16_t raw16 =
            *reinterpret_cast<const std::uint16_t*>(code_row + pair * 64 + lane * 2);
        const float scale = gemv_scale(scale_row + pair * 4 + (lane >> 4) * 2);
        const std::uint32_t b0 = raw16 & 0xFFu;
        const std::uint32_t b1 = (raw16 >> 8) & 0xFFu;
        const std::uint32_t lo0 = 0x64006400u | (b0 & 0x03u) | ((b0 & 0x0Cu) << 14);
        const std::uint32_t hi0 = 0x64006400u | ((b0 >> 4) & 0x03u) | ((b0 & 0xC0u) << 10);
        const std::uint32_t lo1 = 0x64006400u | (b1 & 0x03u) | ((b1 & 0x0Cu) << 14);
        const std::uint32_t hi1 = 0x64006400u | ((b1 >> 4) & 0x03u) | ((b1 & 0xC0u) << 10);
        const half2 k1025 = __float2half2_rn(1025.0F);
        const __half2 w01 = __hsub2(*reinterpret_cast<const half2*>(&lo0), k1025);
        const __half2 w23 = __hsub2(*reinterpret_cast<const half2*>(&hi0), k1025);
        const __half2 w45 = __hsub2(*reinterpret_cast<const half2*>(&lo1), k1025);
        const __half2 w67 = __hsub2(*reinterpret_cast<const half2*>(&hi1), k1025);
        const __half2 a01 = *reinterpret_cast<const __half2*>(&act.x);
        const __half2 a23 = *reinterpret_cast<const __half2*>(&act.y);
        const __half2 a45 = *reinterpret_cast<const __half2*>(&act.z);
        const __half2 a67 = *reinterpret_cast<const __half2*>(&act.w);
        const float2 p0 = __half22float2(__hfma2(w01, a01, __hmul2(w23, a23)));
        const float2 p1 = __half22float2(__hfma2(w45, a45, __hmul2(w67, a67)));
        accumulator = fmaf(scale, (p0.x + p0.y) + (p1.x + p1.y), accumulator);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) {
        out[static_cast<std::int64_t>(0) * out_row_stride + warp] = __float2bfloat16_rn(accumulator);
    }
}

// Warp-staged decode GEMV, for the decode step and the MTP K = 1 verify (T = 1..2; measured, the
// band above it loses -- see the launcher). The shipped kernel reads one code byte per lane per
// group, so eight groups cost the warp eight 32-byte loads, each a separate DRAM latency sitting on
// the decode's dependency chain, plus one 16-bit scale load each. This variant stages eight groups
// at a time: one 64-bit load per lane (256 contiguous bytes per warp, eight sectors in a single
// instruction) into a private per-warp shared buffer, then eight shared-memory byte reads, and the
// block's eight scales come from one broadcast 128-bit load instead of eight 16-bit ones. The next
// block's span is issued before the current block's inner loop, so its DRAM latency overlaps eight
// groups of decode and dot.
//
// Same bytes, same lane->k mapping, same decode, same accumulation order as the tile kernel, so the
// output is bit-identical at every kT -- which is why this can be judged on a plain text comparison
// where the wide-lane variant needed a PPL run.
//
// The eight warps of a block keep private 256-byte buffers, so no __syncthreads() is involved: two
// __syncwarp() per eight groups is enough. That matters, because the one fusion this tree has tried
// (norm+rotation, bit-identical) lost 2.0-2.6% to a block barrier.
//
// Requires groups_per_row % 8 == 0; every width in this model qualifies (40/48/80/136).
//
// kAddResidual folds the layer's residual add into this producer's epilogue; see gemv_store() for
// why the result stays bit-identical to the composed (GEMM + ops::residual_add) route.
template <int kT, int kMinBlocks = 1, int kDepth = 1, bool kAddResidual = false>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32, kMinBlocks)
void ternary_pq2_gemv_stage_kernel(const __nv_bfloat16* __restrict__ x,
                                   const std::uint8_t* __restrict__ codes,
                                   const std::uint8_t* __restrict__ scales,
                                   __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                   std::int32_t groups_per_row, std::int32_t tokens,
                                   std::int32_t out_row_stride,
                                   const __nv_bfloat16* residual = nullptr) {
    static_assert(kT >= 1 && kT <= 5, "the staged kernel serves the decode and MTP verify band");
    __shared__ std::uint8_t stage[kGemvWarpsPerBlock][8 * kGemvCodeBytesPerGroup];
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp_local = static_cast<int>(threadIdx.x) >> 5;
    const int warp = static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + warp_local;
    if (warp >= rows) { return; }
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;
    const std::int64_t token_stride =
        static_cast<std::int64_t>(groups_per_row) * kGemvGroupK;
    const __nv_bfloat16* token_x[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) {
        token_x[t] = x + static_cast<std::int64_t>(t) * token_stride;
    }
    float accumulator[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { accumulator[t] = 0.0F; }

    const int steps = groups_per_row >> 3;
    // Software pipeline: the next block's 256-byte code span is issued BEFORE this block's inner
    // loop, so its DRAM latency overlaps eight groups of decode and dot instead of stalling at the
    // top of the next iteration.
    const std::uint8_t* blk_ptr = code_row;
    uint2 cw = *reinterpret_cast<const uint2*>(blk_ptr + lane * 8);
    // kDepth = 2 keeps one more block's span in flight: the load for block b+2 is issued at the top
    // of block b, so a DRAM latency that overruns one block's worth of decode (about 160 issue
    // slots, ~800 ns at eight warps per scheduler) still has a second block of slack.
    uint2 cw2 = cw;
    if constexpr (kDepth >= 2) {
        if (steps > 1) { cw2 = *reinterpret_cast<const uint2*>(blk_ptr + 256 + lane * 8); }
    }
    for (int blk = 0; blk < steps; ++blk) {
        *reinterpret_cast<uint2*>(&stage[warp_local][lane * 8]) = cw;
        const uint4 sc = *reinterpret_cast<const uint4*>(scale_row + blk * 16);
        blk_ptr += 256;
        uint2 cw_next = cw;
        if constexpr (kDepth >= 2) {
            cw_next = cw2;
            if (blk + 2 < steps) { cw2 = *reinterpret_cast<const uint2*>(blk_ptr + 256 + lane * 8); }
        } else {
            if (blk + 1 < steps) { cw_next = *reinterpret_cast<const uint2*>(blk_ptr + lane * 8); }
        }
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint32_t raw = stage[warp_local][i * kGemvCodeBytesPerGroup + lane];
            const float scale = __half2float(
                __ushort_as_half(reinterpret_cast<const std::uint16_t*>(&sc)[i]));
            const std::uint32_t lo_bits =
                0x64006400u | (raw & 0x03u) | ((raw & 0x0Cu) << 14);
            const std::uint32_t hi_bits =
                0x64006400u | ((raw >> 4) & 0x03u) | ((raw & 0xC0u) << 10);
            const half2 k1025 = __float2half2_rn(1025.0F);
            const __half2 w01 = __hsub2(*reinterpret_cast<const half2*>(&lo_bits), k1025);
            const __half2 w23 = __hsub2(*reinterpret_cast<const half2*>(&hi_bits), k1025);
#pragma unroll
            for (int t = 0; t < kT; ++t) {
                if (t < tokens) {
                    const uint2 packed = *reinterpret_cast<const uint2*>(
                        token_x[t] + static_cast<std::int64_t>(blk * 8 + i) * kGemvGroupK +
                        lane * 4);
                    const __half2 a01 = *reinterpret_cast<const __half2*>(&packed.x);
                    const __half2 a23 = *reinterpret_cast<const __half2*>(&packed.y);
                    const float2 p = __half22float2(__hfma2(w01, a01, __hmul2(w23, a23)));
                    accumulator[t] = fmaf(scale, p.x + p.y, accumulator[t]);
                }
            }
        }
        cw = cw_next;
        __syncwarp();
    }
#pragma unroll
    for (int t = 0; t < kT; ++t) {
        float value = accumulator[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0 && t < tokens) {
            gemv_store<kAddResidual>(out, residual,
                                     static_cast<std::int64_t>(t) * out_row_stride + warp, value);
        }
    }
}

// CTA-staged activation variant of the staged decode kernel -- MEASURED, REJECTED, KEPT AS THE
// RECORD. NINFER_TERNARY_STAGE_ACT=1 selects it; default off, and it should stay off.
//
// The shipped staged kernel gives each warp a PRIVATE 256-byte code buffer -- deliberately, so that
// no block barrier is ever needed -- but it reads the ACTIVATION straight from global once per
// output warp, and all eight warps of a block read exactly the same bytes, because the k walk does
// not depend on the warp. This variant stages one eight-group span of the activation per token in
// shared memory once for the whole block:
//
//   shipped : per warp per block per token   8 x LDG.64 (the same 2 KB, eight times over)
//   this one: per thread per block per token 1 x LDG.64 + 1 x STS.64, then 8 x LDS.64 per warp
//
// WHY IT WAS WORTH BUILDING. Fitting the shipped kernel's own per-call time at T = 1 and T = 2
// gives call(T) ~= 60.1 + 35.0*T us on the 34816x5120 shape, i.e. the activation side is 37% of a
// T = 1 call and 0.5-0.6 of a weight pass per token. The two mechanisms that could pay for that
// without a barrier had both been tried and rejected: widening the lane (TILE_WIDE1, -1.3%) does not
// reduce the BYTES, and blocking rows (kR per warp) does reduce them but costs registers. Shared
// memory is the third way and the only one that leaves the lane mapping, the register budget and the
// arithmetic alone.
//
// MEASURED, same batch, interleaved arms, two repetitions (NINFER_TERNARY_STAGE_ACT=1 against unset):
//
//   T = 1 real_task   55.2/55.2 -> 49.7/49.6   (-10.0%)
//   T = 1 real_code   53.3/53.3 -> 48.1/48.1   ( -9.8%)
//   T = 1 prose4k     51.5/51.5 -> 46.7/46.7   ( -9.3%)
//   MTP K = 1 (T=2)   64.4/64.4 -> 58.2/58.2   ( -9.6%)
//   MTP K = 2 (T=3)   54.9/54.8 -> 54.9/54.8   (  0.0% -- outside the band, the control)
//   lookup10 (T=16)  237.4/237.7 -> 237.7/237.8 ( 0.0% -- outside the band, the control)
//
// WHAT THE NEGATIVE PROVES. The two untouched arms pin the gate, and the ~10% loss says the eight
// duplicate reads were already being served: the warps of a block walk the same lines within a few
// hundred cycles of each other, so they were L1 hits, and this variant traded an L1 hit for an ST S
// plus a barrier while the LDS throughput is no better than L1's. The activation side of this kernel
// is therefore L1-served and latency/issue-bound, NOT L2 or DRAM traffic -- which is exactly the
// case where staging cannot help. It also means the third and last barrier-free mechanism for the
// per-token activation cost is now closed: widening (TILE_WIDE1) does not cut bytes, row blocking
// (kR) cuts bytes but costs registers, and staging cuts bytes but costs instructions and a barrier.
//
// BIT-IDENTICAL, and that held: the 96-token greedy md5 against /root/wt/base.out is IDENTICAL with
// the arm on and off, and the generated text is byte-identical on all twelve run pairs across
// real_task / real_code / prose4k / MTP K=1 / MTP K=2 / lookup10. 26,466 launches of this kernel in
// the tracing run and zero of the shipped one, so the arm really was the one running.
//
// Every warp must reach both barriers, so the shipped kernel's early return for `warp >= rows`
// becomes a predicate here.
template <int kT, int kMinBlocks = 1, bool kAddResidual = false>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32, kMinBlocks)
void ternary_pq2_gemv_stage_act_kernel(const __nv_bfloat16* __restrict__ x,
                                       const std::uint8_t* __restrict__ codes,
                                       const std::uint8_t* __restrict__ scales,
                                       __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                       std::int32_t groups_per_row, std::int32_t tokens,
                                       std::int32_t out_row_stride,
                                       const __nv_bfloat16* residual = nullptr) {
    static_assert(kT >= 1 && kT <= 2, "the CTA-staged variant serves the decode band only");
    constexpr int kBlockGroups = 8;
    constexpr int kBlockK      = kBlockGroups * kGemvGroupK;
    constexpr int kBlockBytes  = kBlockK * 2; // fp16 activation bytes, per token per block
    __shared__ std::uint8_t stage[kGemvWarpsPerBlock][kBlockGroups * kGemvCodeBytesPerGroup];
    __shared__ __align__(16) std::uint8_t act[kT][kBlockBytes];

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp_local = static_cast<int>(threadIdx.x) >> 5;
    const int warp       = static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + warp_local;
    const bool live      = warp < rows; // dead warps still take part in both barriers below

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;
    const std::int64_t token_stride =
        static_cast<std::int64_t>(groups_per_row) * kGemvGroupK;
    const __nv_bfloat16* token_x[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { token_x[t] = x + static_cast<std::int64_t>(t) * token_stride; }

    float accumulator[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { accumulator[t] = 0.0f; }

    const int steps             = groups_per_row >> 3;
    const std::uint8_t* blk_ptr = code_row;
    uint2 cw                    = make_uint2(0u, 0u);
    if (live) { cw = *reinterpret_cast<const uint2*>(blk_ptr + lane * 8); }

    for (int blk = 0; blk < steps; ++blk) {
        // Fill this block's activation span for every live token, once for the whole block. Thread
        // tid owns bytes [tid*8, tid*8+8) of the span; that byte range is exactly the uint2 the
        // shipped kernel loads for group (tid>>5), lane (tid&31), which is what makes this a copy.
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t < tokens) {
                *reinterpret_cast<uint2*>(&act[t][threadIdx.x * 8]) =
                    *reinterpret_cast<const uint2*>(token_x[t] + blk * kBlockK +
                                                    threadIdx.x * 4);
            }
        }
        __syncthreads();
        *reinterpret_cast<uint2*>(&stage[warp_local][lane * 8]) = cw;
        uint4 sc = make_uint4(0u, 0u, 0u, 0u);
        if (live) { sc = *reinterpret_cast<const uint4*>(scale_row + blk * 16); }
        blk_ptr += kGemvWarpsPerBlock * kGemvCodeBytesPerGroup;
        uint2 cw_next = cw;
        if (live && blk + 1 < steps) {
            cw_next = *reinterpret_cast<const uint2*>(blk_ptr + lane * 8);
        }
        __syncwarp();
#pragma unroll
        for (int i = 0; i < kBlockGroups; ++i) {
            const std::uint32_t raw = stage[warp_local][i * kGemvCodeBytesPerGroup + lane];
            const float scale = __half2float(
                __ushort_as_half(reinterpret_cast<const std::uint16_t*>(&sc)[i]));
            const std::uint32_t lo_bits =
                0x64006400u | (raw & 0x03u) | ((raw & 0x0Cu) << 14);
            const std::uint32_t hi_bits =
                0x64006400u | ((raw >> 4) & 0x03u) | ((raw & 0xC0u) << 10);
            const half2 k1025 = __float2half2_rn(1025.0F);
            const __half2 w01 = __hsub2(*reinterpret_cast<const half2*>(&lo_bits), k1025);
            const __half2 w23 = __hsub2(*reinterpret_cast<const half2*>(&hi_bits), k1025);
#pragma unroll
            for (int t = 0; t < kT; ++t) {
                if (t < tokens) {
                    const uint2 packed =
                        *reinterpret_cast<const uint2*>(&act[t][i * 256 + lane * 8]);
                    const __half2 a01 = *reinterpret_cast<const __half2*>(&packed.x);
                    const __half2 a23 = *reinterpret_cast<const __half2*>(&packed.y);
                    const float2 p = __half22float2(__hfma2(w01, a01, __hmul2(w23, a23)));
                    accumulator[t] = fmaf(scale, p.x + p.y, accumulator[t]);
                }
            }
        }
        cw = cw_next;
        // Stronger than the shipped kernel's trailing __syncwarp(): the next iteration's fill must
        // not overwrite `act` while any warp is still reading it.
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < kT; ++t) {
        float value = accumulator[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (live && lane == 0 && t < tokens) {
            gemv_store<kAddResidual>(out, residual,
                                     static_cast<std::int64_t>(t) * out_row_stride + warp, value);
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
