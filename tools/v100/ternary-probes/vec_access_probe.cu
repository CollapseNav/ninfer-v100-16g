// vec_probe4.cu -- A/D said the weight loads AND the whole IQ2_XS decode stream at 768 GB/s (85% of
// peak) on their own, so the real kernel's 285 GB/s (31%) comes from something the probe has not
// reproduced. The obvious candidate is the activation operand: the real kernel re-reads T x 40 bytes of
// quantized activation per lane per iteration -- 6.5x the weight bytes -- from L1, through the same pipe
// the weight loads use.
//
// E adds exactly that traffic.

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kRowsPerThread = 2;
constexpr int kIters   = 5;
constexpr int kBlockBytes = 74;
constexpr int kRowBytes   = (5120 / 256) * 74;
constexpr int kK = 5120;
constexpr int kSlices = kK / 32;          // 160
constexpr int kT = 3;                     // the deployment's verify width

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; } } while (0)

__device__ __forceinline__ std::uint16_t ld16(const std::uint8_t* p) { return __ldg(reinterpret_cast<const std::uint16_t*>(p)); }
__device__ __forceinline__ std::uint8_t  ld8 (const std::uint8_t* p) { return __ldg(p); }
__device__ __forceinline__ std::uint32_t ld32a2(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(ld16(p)) | (static_cast<std::uint32_t>(ld16(p + 2)) << 16);
}
__device__ __forceinline__ int negate_bytes(std::uint32_t g, std::uint32_t bits) {
    const std::uint32_t ones = ((bits & 0xF) * 0x00204081u) & 0x01010101u;
    return int((g ^ (ones * 0xFFu)) + ones);
}
__device__ __forceinline__ std::uint32_t ksigns(std::uint32_t v) {
    return v ^ ((__popc(v) & 1) << 7);
}

// D: the decode, activations held in registers.
__global__ void __launch_bounds__(kThreads, 4) probe_decode(
    const std::uint8_t* __restrict__ w, int rows, float* __restrict__ sink) {
    __shared__ __align__(16) std::uint32_t table[1024];
    for (int i = threadIdx.x; i < 1024; i += kThreads) {
        table[i] = __ldg(reinterpret_cast<const std::uint32_t*>(w) + i);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int u = lane & 7, grp = lane >> 3;
    int a[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { a[i] = (lane * 31 + i * 7) & 0xFF; }
    const uint2* grid = reinterpret_cast<const uint2*>(table);
    float acc = 0.0f;
    for (int row = (blockIdx.x * kWarps + warp) * kRowsPerThread; row < rows;
         row += gridDim.x * kWarps * kRowsPerThread) {
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            const std::uint8_t* base = w + static_cast<std::int64_t>(row + r) * kRowBytes;
#pragma unroll
            for (int it = 0; it < kIters; ++it) {
                const std::uint8_t* b = base + grp * kBlockBytes + it * 4 * kBlockBytes;
                const float d = __half2float(*reinterpret_cast<const half*>(b));
                const std::uint32_t sc = ld8(b + 66 + u);
                const std::uint32_t q01 = ld32a2(b + 2 + 8 * u), q23 = ld32a2(b + 6 + 8 * u);
                int w0[8];
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const std::uint32_t q = ((l < 2 ? q01 : q23) >> (16 * (l & 1))) & 0xFFFFu;
                    const uint2 g = grid[q & 511];
                    const std::uint32_t sg = ksigns(q >> 9);
                    w0[2 * l]     = negate_bytes(g.x, sg);
                    w0[2 * l + 1] = negate_bytes(g.y, sg >> 4);
                }
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) { s0 = __dp4a(w0[i], a[i], s0); }
#pragma unroll
                for (int i = 0; i < 4; ++i) { s1 = __dp4a(w0[4 + i], a[4 + i], s1); }
                const float f0 = d * (0.5f + float(sc & 0xF)) * 0.25f;
                const float f1 = d * (0.5f + float(sc >> 4)) * 0.25f;
                acc += f0 * static_cast<float>(s0) + f1 * static_cast<float>(s1);
            }
        }
    }
    sink[threadIdx.x] = acc;
}

// E: D plus the activation traffic -- T x (32 bytes of q8_1 + an 8-byte float2 scale) per lane per
// iteration, read from a 15 KB buffer the way the real kernel reads p.qs / p.ds.
__global__ void __launch_bounds__(kThreads, 4) probe_decode_act(
    const std::uint8_t* __restrict__ w, int rows, float* __restrict__ sink,
    const std::uint8_t* __restrict__ qs, const float2* __restrict__ ds) {
    __shared__ __align__(16) std::uint32_t table[1024];
    for (int i = threadIdx.x; i < 1024; i += kThreads) {
        table[i] = __ldg(reinterpret_cast<const std::uint32_t*>(w) + i);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int u = lane & 7, grp = lane >> 3;
    const uint2* grid = reinterpret_cast<const uint2*>(table);
    float acc = 0.0f;
    for (int row = (blockIdx.x * kWarps + warp) * kRowsPerThread; row < rows;
         row += gridDim.x * kWarps * kRowsPerThread) {
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            const std::uint8_t* base = w + static_cast<std::int64_t>(row + r) * kRowBytes;
#pragma unroll
            for (int it = 0; it < kIters; ++it) {
                const int s = lane + 32 * it;
                int a[kT][8];
                float d[kT];
#pragma unroll
                for (int j = 0; j < kT; ++j) {
                    const float2 dsv = __ldg(&ds[static_cast<std::int64_t>(j) * kSlices + s]);
                    d[j] = dsv.x;
                    const int4 v0 = __ldg(reinterpret_cast<const int4*>(qs + static_cast<std::int64_t>(j) * kK + 32 * s));
                    const int4 v1 = __ldg(reinterpret_cast<const int4*>(qs + static_cast<std::int64_t>(j) * kK + 32 * s) + 1);
                    a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                    a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
                }
                const std::uint8_t* b = base + grp * kBlockBytes + it * 4 * kBlockBytes;
                const std::uint32_t sc = ld8(b + 66 + u);
                const std::uint32_t q01 = ld32a2(b + 2 + 8 * u), q23 = ld32a2(b + 6 + 8 * u);
                int w0[8];
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const std::uint32_t q = ((l < 2 ? q01 : q23) >> (16 * (l & 1))) & 0xFFFFu;
                    const uint2 g = grid[q & 511];
                    const std::uint32_t sg = ksigns(q >> 9);
                    w0[2 * l]     = negate_bytes(g.x, sg);
                    w0[2 * l + 1] = negate_bytes(g.y, sg >> 4);
                }
#pragma unroll
                for (int j = 0; j < kT; ++j) {
                    int s0 = 0, s1 = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) { s0 = __dp4a(w0[i], a[j][i], s0); }
#pragma unroll
                    for (int i = 0; i < 4; ++i) { s1 = __dp4a(w0[4 + i], a[j][4 + i], s1); }
                    acc += d[j] * static_cast<float>(s0 + s1);
                }
            }
        }
    }
    sink[threadIdx.x] = acc;
}

template <typename K, typename... Args>
static int run(const char* name, K kernel, const std::uint8_t* w, int rows, float* sink, int blocks,
               int reps, Args... extra) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    kernel<<<blocks, kThreads>>>(w, rows, sink, extra...);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) { kernel<<<blocks, kThreads>>>(w, rows, sink, extra...); }
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, a, b));
    const double bytes = static_cast<double>(rows) * kRowBytes * reps;
    std::printf("  %-34s %8.1f ms  %7.1f GB/s useful\n", name, ms, bytes / (ms * 1e-3) / 1e9);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return 0;
}

int main() {
    const int rows = 262144;
    const std::size_t bytes = static_cast<std::size_t>(rows) * kRowBytes;
    std::uint8_t* w = nullptr;
    float* sink = nullptr;
    std::uint8_t* qs = nullptr;
    float2* ds = nullptr;
    CK(cudaMalloc(&w, bytes));
    CK(cudaMalloc(&sink, kThreads * sizeof(float)));
    CK(cudaMalloc(&qs, static_cast<std::size_t>(kT) * kK));
    CK(cudaMalloc(&ds, static_cast<std::size_t>(kT) * kSlices * sizeof(float2)));
    std::uint8_t* host = new std::uint8_t[bytes];
    std::uint32_t s = 12345u;
    for (std::size_t i = 0; i < bytes; ++i) { s = s * 1664525u + 1013904223u; host[i] = std::uint8_t(s >> 24); }
    CK(cudaMemcpy(w, host, bytes, cudaMemcpyHostToDevice));
    delete[] host;
    CK(cudaMemset(qs, 0x37, static_cast<std::size_t>(kT) * kK));
    CK(cudaMemset(ds, 0, static_cast<std::size_t>(kT) * kSlices * sizeof(float2)));
    int dev = 0, sms = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const int groups = (rows + kWarps * kRowsPerThread - 1) / (kWarps * kRowsPerThread);
    const int blocks = groups < sms * 4 ? groups : sms * 4;
    std::printf("device %d, %d SMs, %.0f MB random weights, grid %d, T=%d\n\n", dev, sms, bytes / 1e6,
                blocks, kT);
    if (run("D decode, activations in regs", probe_decode, w, rows, sink, blocks, 20)) return 1;
    if (run("E decode + activation loads", probe_decode_act, w, rows, sink, blocks, 20, qs, ds)) return 1;
    cudaFree(w); cudaFree(sink); cudaFree(qs); cudaFree(ds);
    return 0;
}
