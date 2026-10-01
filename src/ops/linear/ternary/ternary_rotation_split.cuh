// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#pragma once

// Second attempt at the fold rotation's transform time: FOUR warps per D1024 instead of one.
//
// Why: `NINFER_TERNARY_ROTATE_PROBE=copy` measures the launch plus the read/write at ~3.0 us per
// call and the real kernel at ~7.5 us, so ~4.5 us of each of the 258 per-token calls is the
// transform itself. The one-warp kernel pays for it serially: every lane owns 32 elements at stride
// 32, so element (r, lane) is 64 B from its neighbour and the x read is 32 separate 2-byte
// instructions -- plus 32 more for the signs -- before ~480 arithmetic/shuffle instructions run on
// 5 warps of a card with 80 SMs.
//
// The new ownership writes the D1024 index as
//     i = 256*unit + 8*lane + j        (unit 0..3, lane 0..31, j 0..7)
// i.e. global bits 0-2 = j, 3-7 = lane, 8-9 = unit. A lane then owns EIGHT CONTIGUOUS elements, so
// the whole x read is one uint4 (16 B) and the signs are two float4 -- 3 loads instead of 64 -- and
// each warp only has to run its own quarter of the butterfly (~80 ops + 40 shuffles) before the
// four units are joined.
//
// BIT-IDENTICAL, not merely equivalent: a Sylvester H1024 is the product of ten pairwise stages, one
// per index bit, and stages on disjoint bit sets commute. The one-warp kernel executes them in
// global bit order 0..9 (shuffles with stride 1,2,4,8,16 cover lane bits = global bits 0-4, then the
// in-register spans 1,2,4,8,16 cover r bits = global bits 5-9). This kernel executes the SAME ten
// stages in the SAME global order, each with the same `low+high` / `low-high` convention and the
// same `__fadd_rn`/`__fsub_rn` rounding, on the same pairs of indices -- so every intermediate value
// is bit-identical and so is every output. Bits 8-9 (the unit) are the only ones that leave the
// warp; they are joined through 4 KB of shared memory with one __syncthreads(), with the two stages
// rounded in the same order as the sequential kernel would round them.
//
// Scope: forward only, and only when the load is not permuted (NINFER_TERNARY_GDN_PERM is off by
// default). The inverse/embedding rotation and the permuted path keep the one-warp kernel.

#include "ops/linear/ternary/ternary_rotation_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kSplitWarps    = 4; // warps per D1024 (= number of 256-element units)
constexpr int kSplitThreads  = kSplitWarps * kThreadsPerWarp;
constexpr int kSplitElems    = 8; // elements per lane per unit = 16 B of bf16, one uint4

template <bool kFp16Out>
__device__ __forceinline__ void split_store(void* out, std::int64_t index, float value) {
    if constexpr (kFp16Out) {
        static_cast<__half*>(out)[index] =
            __float2half_rn(__bfloat162float(__float2bfloat16_rn(value)));
    } else {
        static_cast<__nv_bfloat16*>(out)[index] = __float2bfloat16_rn(value);
    }
}

// The split transform itself, factored out so the fused norm+rotate kernel runs EXACTLY this
// code on exactly the same values: same uint4/x and float4/signs loads, same ten stages in
// global bit order 0..9, same __fadd_rn/__fsub_rn rounding, same 2^-5 normalizer -- so both
// callers are bit-identical to each other and to the one-warp kernel. `exchange` is the
// caller's shared staging for the two stages that leave the warp (global bits 8-9); `role` is
// the calling warp's position 0..3 within the four-warp group that owns one D1024.
template <bool kFp16Out>
__device__ __forceinline__ void split_transform_unit(
    const __nv_bfloat16* x_ptr, void* out, const float* signs_ptr,
    float (&exchange)[kSplitWarps][32][kSplitElems], int lane, int role,
    std::int64_t out_base) {
    constexpr unsigned FullMask = 0xffffffffu;
    const int unit = role;
    // 8 bf16 in 16 B, 8 fp32 signs in 32 B: three vector loads where the one-warp kernel needs 64
    // scalar ones. Both addresses are 16 B aligned by construction (unit*256 and lane*8).
    const uint4 raw_x   = *reinterpret_cast<const uint4*>(x_ptr);
    const float4 raw_s0 = *reinterpret_cast<const float4*>(signs_ptr);
    const float4 raw_s1 = *reinterpret_cast<const float4*>(signs_ptr + 4);

    const unsigned halves[4] = {raw_x.x, raw_x.y, raw_x.z, raw_x.w};
    const float sgn[8]       = {raw_s0.x, raw_s0.y, raw_s0.z, raw_s0.w,
                                raw_s1.x, raw_s1.y, raw_s1.z, raw_s1.w};

    float values[kSplitElems];
#pragma unroll
    for (int j = 0; j < kSplitElems; ++j) {
        // bf16 is the top 16 bits of the fp32 value: this is exactly __bfloat162float().
        const unsigned short bits = static_cast<unsigned short>(halves[j >> 1] >> (16 * (j & 1)));
        const float value         = __uint_as_float(static_cast<unsigned>(bits) << 16);
        values[j]                 = __fmul_rn(value, sgn[j]); // signs apply before the transform
    }

    // Global bits 0..2 (the contiguous j index): in-register butterfly, spans 1, 2, 4 -- the same
    // low+high / low-high loop the one-warp kernel uses for its r spans.
#pragma unroll
    for (int span = 1; span <= 4; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 8; base += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
                const float low              = values[base + offset];
                const float high             = values[base + offset + span];
                values[base + offset]        = __fadd_rn(low, high);
                values[base + offset + span] = __fsub_rn(low, high);
            }
        }
    }

    // Global bits 3..7 (the lane index): five shuffle stages with stride 1,2,4,8,16, i.e. lane
    // bits 0..4, which are global bits 3..7 because a lane sits at 8*lane.
#pragma unroll
    for (int bit = 0, stride = 1; bit < 5; ++bit, stride <<= 1) {
        const int high = (lane >> bit) & 1;
#pragma unroll
        for (int j = 0; j < kSplitElems; ++j) {
            const float value = values[j];
            const float peer  = __shfl_xor_sync(FullMask, value, stride);
            values[j]         = (high == 0) ? __fadd_rn(value, peer) : __fsub_rn(peer, value);
        }
    }

    // Global bits 8..9 (the unit index): the only stages that leave the warp. Every warp publishes
    // its 8 values, then rounds stage bit8 (pair unit^1) and stage bit9 (pair unit^2) in that
    // order, reading all four raw values so both intermediates are computed here and rounded the
    // same way the sequential kernel would round them.
    #pragma unroll
    for (int j = 0; j < kSplitElems; ++j) { exchange[unit][lane][j] = values[j]; }
    __syncthreads();

    const float* const self  = exchange[unit][lane];
    const float* const pair1 = exchange[unit ^ 1][lane];
    const float* const pair2 = exchange[unit ^ 2][lane];
    const float* const pair3 = exchange[unit ^ 3][lane];
#pragma unroll
    for (int j = 0; j < kSplitElems; ++j) {
        const float bit8_self = ((unit & 1) == 0) ? __fadd_rn(self[j], pair1[j])
                                                  : __fsub_rn(pair1[j], self[j]);
        const float bit8_next = ((unit & 1) == 0) ? __fadd_rn(pair2[j], pair3[j])
                                                  : __fsub_rn(pair3[j], pair2[j]);
        values[j] = ((unit & 2) == 0) ? __fadd_rn(bit8_self, bit8_next)
                                      : __fsub_rn(bit8_next, bit8_self);
    }

    // 2^-5 = 1/sqrt(1024), the same normalizer as the one-warp kernel.
#pragma unroll
    for (int j = 0; j < kSplitElems; ++j) {
        split_store<kFp16Out>(out, out_base + j, __fmul_rn(values[j], 0x1p-5f));
    }
}

template <bool kFp16Out = false>
__global__ void ternary_rotate_bf16_split_kernel(const __nv_bfloat16* __restrict__ x,
                                                 void* __restrict__ out,
                                                 const float* __restrict__ signs, int n_blk, int k,
                                                 int tokens) {
    constexpr unsigned FullMask = 0xffffffffu;

    const int lane   = static_cast<int>(threadIdx.x) & (kThreadsPerWarp - 1);
    const int unit   = static_cast<int>(threadIdx.x) >> 5; // 0..3 -> global bits 8..9
    const int blocks = k >> 10;                            // D1024 blocks per token
    const int token  = static_cast<int>(blockIdx.x) / blocks;
    const int block  = static_cast<int>(blockIdx.x) % blocks;

    const std::int64_t token_base = static_cast<std::int64_t>(token) * k;
    const int unit_base           = (block << 10) + (unit << 8);
    const int element             = unit_base + (lane << 3);

    // x is [k, tokens] TOKEN-major (ne[0] = k is the contiguous axis), so the row is token*k; the
    // signs are per D1024 block and carry no token dimension.
    const __nv_bfloat16* const x_row = x + token_base + element;
    const float* const signs_row =
        signs + static_cast<std::int64_t>(block % n_blk) * kBlockSize + (unit << 8) + (lane << 3);

    __shared__ float exchange[kSplitWarps][32][kSplitElems];
    split_transform_unit<kFp16Out>(x_row, out, signs_row, exchange, lane, unit,
                                   token_base + element);
}

// PROBE: the split-shaped twin of the one-warp copy probe -- same grid, same uint4 read, same
// store path as the real kernel, no sign load and no transform. The default decode path now runs
// four warps per D1024, so the original copy probe prices a shape decode no longer launches; with
// both knobs on, this arm is what "everything except the transform" means. Numerically wrong by
// construction; never a default.
__global__ void ternary_rotate_bf16_split_copy_kernel(const __nv_bfloat16* __restrict__ x,
                                                      void* __restrict__ out, bool fp16_out,
                                                      int k, int tokens) {
    const int lane   = static_cast<int>(threadIdx.x) & (kThreadsPerWarp - 1);
    const int unit   = static_cast<int>(threadIdx.x) >> 5;
    const int blocks = k >> 10;
    const int token  = static_cast<int>(blockIdx.x) / blocks;
    const int block  = static_cast<int>(blockIdx.x) % blocks;

    const std::int64_t token_base = static_cast<std::int64_t>(token) * k;
    const int element             = (block << 10) + (unit << 8) + (lane << 3);

    const uint4 raw     = *reinterpret_cast<const uint4*>(x + token_base + element);
    const unsigned halves[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int j = 0; j < kSplitElems; ++j) {
        const unsigned short bits = static_cast<unsigned short>(halves[j >> 1] >> (16 * (j & 1)));
        const float value         = __uint_as_float(static_cast<unsigned>(bits) << 16);
        if (fp16_out) {
            static_cast<__half*>(out)[token_base + element + j] = __float2half_rn(value);
        } else {
            static_cast<__nv_bfloat16*>(out)[token_base + element + j] = __float2bfloat16_rn(value);
        }
    }
}

} // namespace
} // namespace ninfer::ops::detail
