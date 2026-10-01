#pragma once

// Fused-dequant PQ2_0_G128 (2-bit ternary) x activation GEMM on Volta tensor cores,
// quadpair-split-N form (sm_70 only). Fifth sibling of q4/w8/fp8/nvfp4_volta_qpn_gemm.cuh: same
// CTA shape, same fragment maps, same SPLITK/NACC knobs, same single-barrier cross-warp K reduce.
// What changes is the decoder, the group geometry (128 k per group, one fp16 scale) and -- the
// simplification this format allows -- the code packing.
//
//   - PQ2 packs four 2-bit codes per byte, i.e. four CONSECUTIVE k. One byte is therefore exactly
//     one mma k=4 slice, already in natural k order. NVFP4 has to permute its activations with two
//     PRMTs (nvfp4_stage_pairs) because its 4-bit decoder emits structural (j, j+4) pairs; here
//     both operands reach the mma in the same natural order and contract the matching k with no
//     permutation and no shared memory.
//   - Decode: the bit pattern 0x6400|n is exactly 1024+n as fp16, so one HADD2 against 1025 turns a
//     two-code field into two exact (code-1) values; the group scale multiplies the result in fp16.
//     This is the same magic-constant decode the fp16 tile GEMV already ships.
//   - Scale cadence: one fp16 scale per 128-k group (row-major, contiguous per row), applied to the
//     decoded weights before the mma, so the epilogue needs no multiply -- matching Q4/NVFP4, whose
//     per-group scale is likewise folded in ahead of the tensor core.
//   - kTiles is the number of 8-token A tiles, so T <= 8 * kTiles. SPLITK warps split K by whole
//     groups; NACC breaks the mma RAW chain by round-robining the four slices of one 16-k unit into
//     independent accumulators. Both are the sibling kernels' generation-2 knobs.

#include "core/device.h"
#include "ops/common/volta_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct TernaryVoltaQpnSchedule {
    static constexpr int kColsPerCta       = 32;  // output rows per CTA (mma N axis)
    static constexpr int kRowsPerTile      = 8;   // tokens per A tile (mma M axis)
    static constexpr int kGroupK           = 128; // k-values per PQ2 group (one fp16 scale)
    static constexpr int kCodeBytesPerGroup = 32; // four 2-bit codes per byte
    static constexpr int kBytesPerUnit     = 4;   // 4 bytes = 16 k = 4 mma slices
    static constexpr int kUnitsPerGroup    = kGroupK / 16; // 8
};

// One code byte -> four (code-1) values as two half2, in natural k order, already multiplied by the
// group scale. Byte layout inside the byte is k, k+1, k+2, k+3 in bit fields 0-1, 2-3, 4-5, 6-7.
__device__ __forceinline__ void pq2_decode_byte(std::uint8_t raw, half2 scale, half2* out) {
    const std::uint32_t lo_bits = 0x64006400u | static_cast<std::uint32_t>(raw & 0x03u) |
                                  (static_cast<std::uint32_t>(raw & 0x0Cu) << 14);
    const std::uint32_t hi_bits = 0x64006400u | static_cast<std::uint32_t>((raw >> 4) & 0x03u) |
                                  (static_cast<std::uint32_t>(raw & 0xC0u) << 10);
    const half2 k1025 = __float2half2_rn(1025.0F);
    out[0] = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&lo_bits), k1025), scale);
    out[1] = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&hi_bits), k1025), scale);
}

// The A operand for one 16-k unit: eight fp16 values per thread in natural k order (two uint4 of
// the activation row), no permutation.
template <class Activation>
__device__ __forceinline__ void pq2_stage_activation(const Activation* xrow, half2* a) {
    const uint4 raw0 = *reinterpret_cast<const uint4*>(xrow);
    const uint4 raw1 = *reinterpret_cast<const uint4*>(xrow + 8);
    const Activation* src0 = reinterpret_cast<const Activation*>(&raw0);
    const Activation* src1 = reinterpret_cast<const Activation*>(&raw1);
    half seq[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        if constexpr (std::is_same_v<Activation, half>) {
            seq[j] = src0[j];
            seq[8 + j] = src1[j];
        } else {
            seq[j] = __float2half(__bfloat162float(src0[j]));
            seq[8 + j] = __float2half(__bfloat162float(src1[j]));
        }
    }
    const auto* pairs = reinterpret_cast<const half2*>(seq);
#pragma unroll
    for (int j = 0; j < 8; ++j) { a[j] = pairs[j]; }
}

// `t` is the live token count; rows beyond it are zeroed so the mma still executes.
template <int kTiles, int SPLITK, int NACC, class OutputPolicy, class Activation>
__global__ __launch_bounds__(
    SPLITK * 32, (kTiles == 1 ? 32 : kTiles == 2 ? 16 : 4) / SPLITK < 1
        ? 1
        : (kTiles == 1 ? 32 : kTiles == 2 ? 16 : 4) / SPLITK)
void ternary_volta_qpn_gemm_kernel(const std::uint8_t* __restrict__ codes,
                                   const std::uint8_t* __restrict__ scales,
                                   const Activation* __restrict__ x, int n, int k, int t,
                                   OutputPolicy output) {
    using S = TernaryVoltaQpnSchedule;

    __shared__ float cs[SPLITK][kTiles * S::kRowsPerTile * S::kColsPerCta];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int qp   = (lane >> 2) & 3;
    const int r    = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);

    const int col     = static_cast<int>(blockIdx.x) * S::kColsPerCta + qp * 8 + r;
    const int good    = col < n;
    const int use_col = good ? col : 0;

    const int groups = k / S::kGroupK;
    const int gq     = groups / SPLITK;
    const int g0     = warp * gq;
    const int gend   = (warp == SPLITK - 1) ? groups : g0 + gq;

    // Row-major code plane: k/4 bytes per output row. Row-major scale plane: one fp16 per group.
    // QPN-prepacked plane (ternary_prepack_qpn): entry (tile, group, lane) holds the whole
    // 32-byte code group of that output row followed by its 2-byte scale, so a warp fetches one
    // contiguous 1 KB block per group instead of 32 scattered rows. Production layout upstream.
    const std::int64_t tile_base = static_cast<std::int64_t>(blockIdx.x) * groups * 32 + lane;

    float c[kTiles][NACC][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int a = 0; a < NACC; ++a) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { c[tile][a][i] = 0.0F; }
        }
    }

    for (int st = g0; st < gend; ++st) {
        const std::int64_t packed_index = tile_base + static_cast<std::int64_t>(st) * 32;
        const half2 sc2 = __half2half2(
            __ushort_as_half(reinterpret_cast<const std::uint16_t*>(scales)[packed_index]));
        const std::uint8_t* cp = codes + packed_index * S::kCodeBytesPerGroup;
#pragma unroll
        for (int u = 0; u < S::kUnitsPerGroup; ++u) {
            const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(
                cp + static_cast<std::int64_t>(u) * S::kBytesPerUnit);
            half2 b[8];
            pq2_decode_byte(static_cast<std::uint8_t>(word & 0xFFu), sc2, &b[0]);
            pq2_decode_byte(static_cast<std::uint8_t>((word >> 8) & 0xFFu), sc2, &b[2]);
            pq2_decode_byte(static_cast<std::uint8_t>((word >> 16) & 0xFFu), sc2, &b[4]);
            pq2_decode_byte(static_cast<std::uint8_t>((word >> 24) & 0xFFu), sc2, &b[6]);
            const unsigned* B = reinterpret_cast<const unsigned*>(b);
            const int kbase   = st * S::kGroupK + u * 16;

#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                const int row = tile * S::kRowsPerTile + r;
                half2 a[8];
                if (row < t) {
                    pq2_stage_activation<Activation>(x + static_cast<std::int64_t>(row) * k + kbase,
                                                     a);
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { a[j] = __half2half2(__ushort_as_half(0)); }
                }
                const unsigned* A = reinterpret_cast<const unsigned*>(a);
                volta_mma_qp_n(c[tile][0 % NACC], A[0], A[1], B[0], B[1]);
                volta_mma_qp_n(c[tile][1 % NACC], A[2], A[3], B[2], B[3]);
                volta_mma_qp_n(c[tile][2 % NACC], A[4], A[5], B[4], B[5]);
                volta_mma_qp_n(c[tile][3 % NACC], A[6], A[7], B[6], B[7]);
            }
        }
    }

#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int a = 1; a < NACC; ++a) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { c[tile][0][i] += c[tile][a][i]; }
        }
    }

    // C map, identical to the sibling kernels.
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][(tile * S::kRowsPerTile + row) * S::kColsPerCta + qp * 8 + cl] = c[tile][0][i];
        }
    }
    __syncthreads(); // the only barrier: cross-warp K reduce

    constexpr int kOut = kTiles * S::kRowsPerTile * S::kColsPerCta;
    for (int e = static_cast<int>(threadIdx.x); e < kOut; e += SPLITK * 32) {
        const int row  = e / S::kColsPerCta;
        const int cl   = e % S::kColsPerCta;
        const int ocol = static_cast<int>(blockIdx.x) * S::kColsPerCta + cl;
        if (row < t && ocol < n) {
            float v = 0.0F;
#pragma unroll
            for (int w = 0; w < SPLITK; ++w) { v += cs[w][e]; }
            output.store(ocol, row, v);
        }
    }
}

#endif // arch guard

} // namespace ninfer::ops::detail
