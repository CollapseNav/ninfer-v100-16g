// Does ternary_pq2_gemv_tile_kernel actually amortise the weights across the verify tile?
//
// The MTP verify is the only route to a 100+ decode rate on this card (v100.md's own round table shows
// their K=1 -> K=3 round growing 23.80 -> 30.20 ms, i.e. +2 draft tokens for +27%), and our round
// measures 2.96x a T=1 token for K=3. The kernel claims to load the code byte and the scale once per
// group and reuse them for all kT tokens, so this measures that claim directly: the same weight, the
// same grid, only the token count changing.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>

#define CK(x)                                                                        \
    do {                                                                             \
        cudaError_t e = (x);                                                         \
        if (e != cudaSuccess) {                                                      \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, \
                        __LINE__);                                                   \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

constexpr int kWarpsPerBlock = 8;
constexpr int kGroupK        = 128;
constexpr int kCodeBytes     = 32;
constexpr int kScaleBytes    = 2;

__device__ __forceinline__ float gemv_scale(const std::uint8_t* scale_ptr) {
    const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(scale_ptr);
    return __half2float(__ushort_as_half(bits));
}

// Verbatim from ternary_rowsplit_gemv.cuh, minus the launch_bounds macro.
template <int kT>
__global__ __launch_bounds__(kWarpsPerBlock * 32)
void ternary_pq2_gemv_tile_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                  std::int32_t groups_per_row, std::int32_t tokens,
                                  std::int32_t out_row_stride) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kCodeBytes;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kScaleBytes;

    float accumulator[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { accumulator[t] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        const std::uint8_t raw = code_row[group * kCodeBytes + lane];
        const float scale = gemv_scale(scale_row + group * kScaleBytes);
        const float weight0 = static_cast<float>(static_cast<int>(raw & 3u) - 1);
        const float weight1 = static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1);
        const float weight2 = static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1);
        const float weight3 = static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1);

        const std::int32_t base = group * kGroupK + lane * 4;
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t < tokens) {
                const __nv_bfloat16* x_token =
                    x + static_cast<std::int64_t>(t) * groups_per_row * kGroupK;
                const float2 low = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base));
                const float2 high = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + 2));
                const float dot = fmaf(weight0, low.x,
                                       fmaf(weight1, low.y, fmaf(weight2, high.x, weight3 * high.y)));
                accumulator[t] = fmaf(scale, dot, accumulator[t]);
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
            out[static_cast<std::int64_t>(t) * out_row_stride + warp] =
                __float2bfloat16_rn(value);
        }
    }
}

// The shipped T=1 kernel, for the ratio that matters: what does a verify tile cost against a plain
// decode step on the same weights?
template <int kUnroll>
__global__ __launch_bounds__(kWarpsPerBlock * 32)
void ternary_pq2_gemv_w_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out, std::int32_t rows,
                               std::int32_t groups_per_row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }
    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kCodeBytes;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kScaleBytes;
    float accumulator = 0.0f;
#pragma unroll(kUnroll)
    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGroupK + lane * 4;
        const float2 low = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base));
        const float2 high = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base + 2));
        const std::uint8_t raw = code_row[group * kCodeBytes + lane];
        const float dot = fmaf(static_cast<float>(static_cast<int>(raw & 3u) - 1), low.x,
                               fmaf(static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1),
                                    low.y,
                                    fmaf(static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1),
                                         high.x,
                                         static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1) *
                                             high.y)));
        accumulator = fmaf(gemv_scale(scale_row + group * kScaleBytes), dot, accumulator);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) { out[warp] = __float2bfloat16_rn(accumulator); }
}

int main(int argc, char** argv) {
    const int rows   = 34816 * 4; // 178 MB of codes: long enough that launch overhead is noise
    const int groups = 40;
    const std::int64_t code_bytes = static_cast<std::int64_t>(rows) * groups * kCodeBytes;
    std::uint8_t* codes = nullptr;
    std::uint8_t* scales = nullptr;
    __nv_bfloat16* x = nullptr;
    __nv_bfloat16* out = nullptr;
    CK(cudaMalloc(&codes, code_bytes));
    CK(cudaMalloc(&scales, static_cast<std::int64_t>(rows) * groups * kScaleBytes));
    CK(cudaMalloc(&x, static_cast<std::int64_t>(8) * groups * kGroupK * 2));
    CK(cudaMalloc(&out, static_cast<std::int64_t>(8) * rows * 2));
    CK(cudaMemset(codes, 0x55, code_bytes));
    CK(cudaMemset(scales, 0x00, static_cast<std::int64_t>(rows) * groups * kScaleBytes));
    CK(cudaMemset(x, 0x3f, static_cast<std::int64_t>(8) * groups * kGroupK * 2));

    const int reps = argc > 1 ? std::atoi(argv[1]) : 40;
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    const dim3 grid(static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock));
    const double mb = code_bytes / 1e6;

    auto timeit = [&](const char* tag, auto&& fn) {
        for (int i = 0; i < 3; ++i) { fn(); }
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(t0));
        for (int i = 0; i < reps; ++i) { fn(); }
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms = 0.0F;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= static_cast<float>(reps);
        std::printf("  %-40s %8.3f ms  %6.1f GB/s\n", tag, ms, mb / (ms * 1e-3) / 1e3);
        return ms;
    };

    std::printf("rows=%d groups=%d  code plane %.0f MB  reps=%d\n", rows, groups, mb, reps);
    const float t1ms = timeit("T=1 GEMV (w_kernel, unroll 8)", [&] {
        ternary_pq2_gemv_w_kernel<8><<<grid, 256>>>(x, codes, scales, out, rows, groups);
    });
    std::printf("  ---- tile kernel, kT = tokens (no guard waste) ----\n");
    float prev = t1ms;
    timeit("tile kT=1 tokens=1", [&] {
        ternary_pq2_gemv_tile_kernel<1><<<grid, 256>>>(x, codes, scales, out, rows, groups, 1, rows);
    });
    const float k2 = timeit("tile kT=2 tokens=2", [&] {
        ternary_pq2_gemv_tile_kernel<2><<<grid, 256>>>(x, codes, scales, out, rows, groups, 2, rows);
    });
    const float k3 = timeit("tile kT=3 tokens=3", [&] {
        ternary_pq2_gemv_tile_kernel<3><<<grid, 256>>>(x, codes, scales, out, rows, groups, 3, rows);
    });
    const float k4 = timeit("tile kT=4 tokens=4", [&] {
        ternary_pq2_gemv_tile_kernel<4><<<grid, 256>>>(x, codes, scales, out, rows, groups, 4, rows);
    });
    const float k4p2 = timeit("tile kT=4 tokens=2 (what MTP K=1 runs)", [&] {
        ternary_pq2_gemv_tile_kernel<4><<<grid, 256>>>(x, codes, scales, out, rows, groups, 2, rows);
    });
    std::printf("  ---- scaling (per token embodied) ----\n");
    std::printf("  M=2 / M=1 : %.2fx  (weight-resident would be ~1.05)\n", k2 / t1ms);
    std::printf("  M=3 / M=1 : %.2fx\n", k3 / t1ms);
    std::printf("  M=4 / M=1 : %.2fx\n", k4 / t1ms);
    std::printf("  M=2 via kT=4 / M=1 : %.2fx\n", k4p2 / t1ms);
    (void)prev;
    CK(cudaDeviceSynchronize());
    return 0;
}
