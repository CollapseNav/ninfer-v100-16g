// reg_probe.cu -- which construct in the real vec kernel's inner loop costs the 30-45 registers that drop
// resident blocks from 4 to 2-3 per SM?
//
// The standalone probe of the same inner loop compiles to <=64 registers with __launch_bounds__(256,4).
// The real kernel, same loop, compiles to 80-109 and loses occupancy. Build the loop up in steps and read
// the register count after each -- compile-only, no GPU needed.
//
//   V0  the real structure: rowp[kRows] array, row-address arithmetic, Fused branch, T=3
//   V1  no rowp[] array (one row at a time, address computed in the loop)
//   V2  V1 without the Fused branch
//   V3  V1 with T=1
//   V4  the probe's lean shape (no rowp, no Fused, T=3, direct addressing)
//
// Build: nvcc -O3 -std=c++17 --generate-code=arch=compute_70,code=[compute_70,sm_70] -cubin reg_probe.cu
//        cuobjdump --dump-resource-usage reg_probe.cubin

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kBlockBytes = 74;
constexpr int kRowBytes   = (5120 / 256) * 74;
constexpr int kK = 5120;
constexpr int kSlices = kK / 32;

__device__ __forceinline__ std::uint16_t ld16(const std::uint8_t* p) { return __ldg(reinterpret_cast<const std::uint16_t*>(p)); }
__device__ __forceinline__ std::uint8_t  ld8 (const std::uint8_t* p) { return __ldg(p); }
__device__ __forceinline__ std::uint32_t ld32a2(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(ld16(p)) | (static_cast<std::uint32_t>(ld16(p + 2)) << 16);
}
__device__ __forceinline__ int negate_bytes(std::uint32_t g, std::uint32_t bits) {
    const std::uint32_t ones = ((bits & 0xF) * 0x00204081u) & 0x01010101u;
    return int((g ^ (ones * 0xFFu)) + ones);
}
__device__ __forceinline__ std::uint32_t ksigns(std::uint32_t v) { return v ^ ((__popc(v) & 1) << 7); }

// the IQ2_XS decode, structurally identical to the real one
__device__ __forceinline__ void decode(const std::uint8_t* b, int u, const uint2* grid, int w0[8]) {
    const std::uint32_t sc = ld8(b + 66 + u);
    const std::uint32_t q01 = ld32a2(b + 2 + 8 * u), q23 = ld32a2(b + 6 + 8 * u);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const std::uint32_t q = ((l < 2 ? q01 : q23) >> (16 * (l & 1))) & 0xFFFFu;
        const uint2 g = grid[q & 511];
        const std::uint32_t sg = ksigns(q >> 9);
        w0[2 * l]     = negate_bytes(g.x, sg);
        w0[2 * l + 1] = negate_bytes(g.y, sg >> 4);
    }
}

template <int T, bool UseRowp, bool Fused>
__global__ void __launch_bounds__(kThreads, 4) variant(
    const std::uint8_t* __restrict__ w, int rows, float* __restrict__ sink,
    const std::uint8_t* __restrict__ qs, const float2* __restrict__ ds, long long row_bytes) {
    __shared__ __align__(16) std::uint32_t table[1024];
    for (int i = threadIdx.x; i < 1024; i += kThreads) { table[i] = __ldg(reinterpret_cast<const std::uint32_t*>(w) + i); }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int u = lane & 7, grp = lane >> 3;
    const std::int64_t lane_offset = std::int64_t(lane / 8) * kBlockBytes;
    const uint2* grid = reinterpret_cast<const uint2*>(table);
    float acc[2][T];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
    const int step = Fused ? 1 : 2;
    for (int row0 = (blockIdx.x * kWarps + warp) * step; row0 < rows; row0 += gridDim.x * kWarps * step) {
        const std::uint8_t* rowp[2];
        if constexpr (UseRowp) {
            if constexpr (Fused) {
                rowp[0] = w + std::int64_t(row0) * row_bytes + lane_offset;
                rowp[1] = w + std::int64_t(rows + row0) * row_bytes + lane_offset;
            } else {
                rowp[0] = w + std::int64_t(min(row0, rows - 1)) * row_bytes + lane_offset;
                rowp[1] = w + std::int64_t(min(row0 + 1, rows - 1)) * row_bytes + lane_offset;
            }
        }
#pragma unroll
        for (int it = 0; it < kK / 1024; ++it) {
            const int s = lane + 32 * it;
            int a[T][8];
            float d[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                d[j] = __ldg(&ds[static_cast<std::int64_t>(j) * kSlices + s]).x;
                const int4 v0 = __ldg(reinterpret_cast<const int4*>(qs + static_cast<std::int64_t>(j) * kK + 32 * s));
                const int4 v1 = __ldg(reinterpret_cast<const int4*>(qs + static_cast<std::int64_t>(j) * kK + 32 * s) + 1);
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const std::uint8_t* b;
                if constexpr (UseRowp) {
                    b = rowp[r] + it * 4 * kBlockBytes;
                } else {
                    const int row = (Fused ? row0 : min(row0 + r, rows - 1));
                    b = w + std::int64_t(Fused ? (r ? rows + row : row) : row) * row_bytes + lane_offset +
                        it * 4 * kBlockBytes;
                }
                int w0[8];
                decode(b, u, grid, w0);
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    int s0 = 0, s1 = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) { s0 = __dp4a(w0[i], a[j][i], s0); }
#pragma unroll
                    for (int i = 0; i < 4; ++i) { s1 = __dp4a(w0[4 + i], a[j][4 + i], s1); }
                    acc[r][j] += d[j] * static_cast<float>(s0 + s1);
                }
            }
        }
    }
    float total = 0.0f;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int j = 0; j < T; ++j) { total += acc[r][j]; }
    sink[threadIdx.x] = total;
}

template __global__ void variant<3, true, false>(const std::uint8_t*, int, float*, const std::uint8_t*, const float2*, long long);
template __global__ void variant<3, false, false>(const std::uint8_t*, int, float*, const std::uint8_t*, const float2*, long long);
template __global__ void variant<3, true, true>(const std::uint8_t*, int, float*, const std::uint8_t*, const float2*, long long);
template __global__ void variant<1, true, false>(const std::uint8_t*, int, float*, const std::uint8_t*, const float2*, long long);
