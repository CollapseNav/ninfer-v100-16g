// Standalone measurement of the PQ2 -> fp16 dequant kernel.
//
// The in-situ probe decomposition says this kernel costs 0.55 s of the 3.83 s prefill, which for the
// 46 GB it moves (5.4 GB of codes read, 40.8 GB of fp16 written) is 84 GB/s -- ten times below the
// card's 900 GB/s. This measures it directly instead of inferring it from a three-way subtraction,
// and separates the read and the write side so the ceiling for each is visible.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(x)                                                                        \
    do {                                                                             \
        cudaError_t e = (x);                                                         \
        if (e != cudaSuccess) {                                                      \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, \
                        __LINE__);                                                   \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

constexpr int kGroupK  = 128;
constexpr int kCodeB   = 32;   // code bytes per group
constexpr int kScaleB  = 2;    // fp16 scale per group
constexpr int kPerByte = 4;    // codes per byte

__forceinline__ __device__ unsigned bits_of(__half2 h) {
    return *reinterpret_cast<unsigned*>(&h);
}

// The current kernel, copied verbatim from ternary_cutlass_sm70.cu.
__global__ void dequant_wide(const std::uint8_t* __restrict__ codes,
                             const std::uint8_t* __restrict__ scales,
                             std::int32_t row0, std::int32_t rows, std::int32_t k,
                             std::int32_t groups, __half* __restrict__ out) {
    constexpr int kPerWord = 4;
    constexpr int kWeights = kPerWord * kPerByte;
    constexpr int kThreadsPerGroup = kCodeB / kPerWord;
    constexpr int kGroupsPerBlock  = 256 / kThreadsPerGroup;

    const std::int32_t g = static_cast<std::int32_t>(blockIdx.x) * kGroupsPerBlock +
                           static_cast<std::int32_t>(threadIdx.x) / kThreadsPerGroup;
    const std::int32_t r = static_cast<std::int32_t>(blockIdx.y);
    if (r >= rows || g >= groups) { return; }
    const std::int32_t bq = (static_cast<std::int32_t>(threadIdx.x) % kThreadsPerGroup) * kPerWord;

    const std::int64_t code_base =
        static_cast<std::int64_t>(row0 + r) * groups * kCodeB + static_cast<std::int64_t>(g) * kCodeB;
    const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(codes + code_base + bq);
    const __half scale = __ushort_as_half(
        reinterpret_cast<const std::uint16_t*>(scales + static_cast<std::int64_t>(row0 + r) *
                                                             groups * kScaleB)[g]);
    const float s = __half2float(scale);
    __half2 p[kWeights / 2];
#pragma unroll
    for (int i = 0; i < kWeights / 2; ++i) {
        const int c0 = static_cast<int>((word >> (4 * i)) & 0x3u);
        const int c1 = static_cast<int>((word >> (4 * i + 2)) & 0x3u);
        p[i] = __floats2half2_rn(static_cast<float>(c0 - 1) * s, static_cast<float>(c1 - 1) * s);
    }
    __half* dst = out + static_cast<std::int64_t>(r) * k + static_cast<std::int64_t>(g) * kGroupK +
                  bq * kPerByte;
    uint4 v0, v1;
    v0.x = bits_of(p[0]); v0.y = bits_of(p[1]); v0.z = bits_of(p[2]); v0.w = bits_of(p[3]);
    v1.x = bits_of(p[4]); v1.y = bits_of(p[5]); v1.z = bits_of(p[6]); v1.w = bits_of(p[7]);
    auto* out4 = reinterpret_cast<uint4*>(dst);
    out4[0] = v0;
    out4[1] = v1;
}

// Write-only control: same store pattern, no code read. Isolates the store ceiling.
__global__ void store_only(__half* __restrict__ out, std::int32_t rows, std::int32_t k,
                           std::int32_t groups) {
    constexpr int kPerWord = 4;
    constexpr int kWeights = kPerWord * kPerByte;
    constexpr int kThreadsPerGroup = kCodeB / kPerWord;
    constexpr int kGroupsPerBlock  = 256 / kThreadsPerGroup;
    const std::int32_t g = static_cast<std::int32_t>(blockIdx.x) * kGroupsPerBlock +
                           static_cast<std::int32_t>(threadIdx.x) / kThreadsPerGroup;
    const std::int32_t r = static_cast<std::int32_t>(blockIdx.y);
    if (r >= rows || g >= groups) { return; }
    const std::int32_t bq = (static_cast<std::int32_t>(threadIdx.x) % kThreadsPerGroup) * kPerWord;
    __half* dst = out + static_cast<std::int64_t>(r) * k + static_cast<std::int64_t>(g) * kGroupK +
                  bq * kPerByte;
    uint4 v;
    v.x = 0x3c003c00u; v.y = v.x; v.z = v.x; v.w = v.x;
    auto* out4 = reinterpret_cast<uint4*>(dst);
    out4[0] = v;
    out4[1] = v;
}

// Read-only control: same code read, one dummy dependent store. Isolates the read ceiling.
__global__ void read_only(const std::uint8_t* __restrict__ codes,
                          const std::uint8_t* __restrict__ scales, std::int32_t rows,
                          std::int32_t k, std::int32_t groups, unsigned* __restrict__ sink) {
    constexpr int kPerWord = 4;
    constexpr int kThreadsPerGroup = kCodeB / kPerWord;
    constexpr int kGroupsPerBlock  = 256 / kThreadsPerGroup;
    const std::int32_t g = static_cast<std::int32_t>(blockIdx.x) * kGroupsPerBlock +
                           static_cast<std::int32_t>(threadIdx.x) / kThreadsPerGroup;
    const std::int32_t r = static_cast<std::int32_t>(blockIdx.y);
    if (r >= rows || g >= groups) { return; }
    const std::int32_t bq = (static_cast<std::int32_t>(threadIdx.x) % kThreadsPerGroup) * kPerWord;
    const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(
        codes + static_cast<std::int64_t>(r) * groups * kCodeB + static_cast<std::int64_t>(g) * kCodeB + bq);
    const __half scale = __ushort_as_half(
        reinterpret_cast<const std::uint16_t*>(scales + static_cast<std::int64_t>(r) * groups * kScaleB)[g]);
    const unsigned acc = word + static_cast<unsigned>(__half_as_ushort(scale));
    if (acc == 0xdeadbeefu) { sink[0] = acc; }
}

// Fixed mapping: one 16-byte store per thread, so a warp's stores are one contiguous run instead of
// 32 half-filled sectors. 16 threads per group, each owning 2 code bytes (8 weights).
__global__ void dequant_v2(const std::uint8_t* __restrict__ codes,
                           const std::uint8_t* __restrict__ scales, std::int32_t rows,
                           std::int32_t k, std::int32_t groups, __half* __restrict__ out) {
    constexpr int kThreadsPerGroup = 16;   // 32 code bytes / 2 per thread
    constexpr int kGroupsPerBlock  = 256 / kThreadsPerGroup;
    const std::int32_t g = static_cast<std::int32_t>(blockIdx.x) * kGroupsPerBlock +
                           static_cast<std::int32_t>(threadIdx.x) / kThreadsPerGroup;
    const std::int32_t r = static_cast<std::int32_t>(blockIdx.y);
    if (r >= rows || g >= groups) { return; }
    const std::int32_t bq = (static_cast<std::int32_t>(threadIdx.x) % kThreadsPerGroup) * 2;

    const std::uint16_t word = *reinterpret_cast<const std::uint16_t*>(
        codes + static_cast<std::int64_t>(r) * groups * kCodeB + static_cast<std::int64_t>(g) * kCodeB + bq);
    const __half scale = __ushort_as_half(
        reinterpret_cast<const std::uint16_t*>(scales + static_cast<std::int64_t>(r) * groups * kScaleB)[g]);
    const float s = __half2float(scale);
    __half2 p[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c0 = static_cast<int>((word >> (4 * i)) & 0x3u);
        const int c1 = static_cast<int>((word >> (4 * i + 2)) & 0x3u);
        p[i] = __floats2half2_rn(static_cast<float>(c0 - 1) * s, static_cast<float>(c1 - 1) * s);
    }
    __half* dst = out + static_cast<std::int64_t>(r) * k + static_cast<std::int64_t>(g) * kGroupK +
                  bq * kPerByte;
    uint4 v;
    v.x = bits_of(p[0]); v.y = bits_of(p[1]); v.z = bits_of(p[2]); v.w = bits_of(p[3]);
    *reinterpret_cast<uint4*>(dst) = v;
}

__global__ void store_only_v2(__half* __restrict__ out, std::int32_t rows, std::int32_t k,
                              std::int32_t groups) {
    constexpr int kThreadsPerGroup = 16;
    constexpr int kGroupsPerBlock  = 256 / kThreadsPerGroup;
    const std::int32_t g = static_cast<std::int32_t>(blockIdx.x) * kGroupsPerBlock +
                           static_cast<std::int32_t>(threadIdx.x) / kThreadsPerGroup;
    const std::int32_t r = static_cast<std::int32_t>(blockIdx.y);
    if (r >= rows || g >= groups) { return; }
    const std::int32_t bq = (static_cast<std::int32_t>(threadIdx.x) % kThreadsPerGroup) * 2;
    __half* dst = out + static_cast<std::int64_t>(r) * k + static_cast<std::int64_t>(g) * kGroupK +
                  bq * kPerByte;
    uint4 v;
    v.x = 0x3c003c00u; v.y = v.x; v.z = v.x; v.w = v.x;
    *reinterpret_cast<uint4*>(dst) = v;
}

template <class F>
void timeit(const char* tag, int nx, int k, double bytes, F fn) {
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    for (int i = 0; i < 2; ++i) { fn(); }
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(t0));
    for (int i = 0; i < 20; ++i) { fn(); }
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms = 0.0F;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= 20.0F;
    std::printf("  %-34s %8.3f ms  %7.1f GB/s  (%.2f GB)\n", tag, ms, bytes / (ms * 1e-3) / 1e9,
                bytes / 1e9);
    (void)nx;
    (void)k;
}

int main() {
    const int k = 5120, n = 34816, groups = k / kGroupK, rows = n;
    std::uint8_t* codes = nullptr;
    std::uint8_t* scales = nullptr;
    __half* out = nullptr;
    unsigned* sink = nullptr;
    CK(cudaMalloc(&codes, static_cast<std::size_t>(rows) * groups * kCodeB));
    CK(cudaMalloc(&scales, static_cast<std::size_t>(rows) * groups * kScaleB));
    CK(cudaMalloc(&out, static_cast<std::size_t>(rows) * k * 2));
    CK(cudaMalloc(&sink, 4));
    CK(cudaMemset(codes, 0x55, static_cast<std::size_t>(rows) * groups * kCodeB));
    CK(cudaMemset(scales, 0x00, static_cast<std::size_t>(rows) * groups * kScaleB));

    const double code_bytes = static_cast<double>(rows) * groups * kCodeB;
    const double scale_bytes = static_cast<double>(rows) * groups * kScaleB;
    const double out_bytes = static_cast<double>(rows) * k * 2;
    std::printf("n=%d k=%d groups=%d  codes=%.2f GB out=%.2f GB\n", n, k, groups, code_bytes / 1e9,
                out_bytes / 1e9);

    constexpr int kThreadsPerGroup = kCodeB / 4;
    const int threads = 256;
    const dim3 grid((groups + 256 / kThreadsPerGroup - 1) / (256 / kThreadsPerGroup),
                    static_cast<unsigned>(rows), 1u);

    timeit("store_only (v1 map)", n, k, out_bytes, [&] { store_only<<<grid, threads>>>(out, rows, k, groups); });
    timeit("store_only (v2 map)", n, k, out_bytes, [&] { store_only_v2<<<grid, threads>>>(out, rows, k, groups); });
    timeit("read_only", n, k, code_bytes + scale_bytes,
           [&] { read_only<<<grid, threads>>>(codes, scales, rows, k, groups, sink); });
    timeit("dequant v1", n, k, code_bytes + scale_bytes + out_bytes,
           [&] { dequant_wide<<<grid, threads>>>(codes, scales, 0, rows, k, groups, out); });
    timeit("dequant v2", n, k, code_bytes + scale_bytes + out_bytes,
           [&] { dequant_v2<<<grid, threads>>>(codes, scales, rows, k, groups, out); });
    CK(cudaDeviceSynchronize());
    std::printf("  grid=(%u,%u) threads=%d blocks=%u\n", grid.x, grid.y, threads,
                grid.x * grid.y);

    // Same work expressed as one thread per 16 bytes of output, to see whether the per-thread work
    // is simply too small for this grid.
    timeit("dequant as pure memset-speed ref", n, k, out_bytes,
           [&] { CK(cudaMemsetAsync(out, 0, static_cast<std::size_t>(rows) * k * 2)); });
    return 0;
}
