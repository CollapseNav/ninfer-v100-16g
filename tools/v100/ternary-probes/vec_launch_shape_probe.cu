// vec_probe6.cu -- the standalone probe reaches 509-580 GB/s with the vec kernel's own inner loop, while
// the kernel achieves about 285 GB/s in situ. Two differences have not been reproduced:
//
//   H  the real decode is ~143 short launches per round, each over a *different* weight tensor, rather
//      than one long stream over one buffer;
//   I  the vec launches are interleaved with attention launches that stream the KV cache.
//
// This probe runs the same inner loop under both conditions, with realistic per-tensor sizes and the
// launch's real grid rule, min(groups, resident * sm_count).
//
// Build: nvcc -O3 -std=c++17 --generate-code=arch=compute_70,code=[compute_70,sm_70] -o vec_probe6 vec_probe6.cu

#include <cstdint>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kRowsPerThread = 2;
constexpr int kIters   = 5;
constexpr int kBlockBytes = 74;
constexpr int kK = 5120;
constexpr int kRowBytes = (kK / 256) * 74;      // 1480
constexpr int kSlices = kK / 32;
constexpr int kT = 3;

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
__device__ __forceinline__ std::uint32_t ksigns(std::uint32_t v) { return v ^ ((__popc(v) & 1) << 7); }

// the vec kernel's inner loop: weight loads + the IQ2_XS decode + T=3 activation loads and dp4a
__global__ void __launch_bounds__(kThreads, 4) vec_like(
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
        for (int it = 0; it < kIters; ++it) {
            const int s = lane + 32 * it;
            int a[kT][8];
            float d[kT];
#pragma unroll
            for (int j = 0; j < kT; ++j) {
                d[j] = __ldg(&ds[static_cast<std::int64_t>(j) * kSlices + s]).x;
                const int4 v0 = __ldg(reinterpret_cast<const int4*>(qs + static_cast<std::int64_t>(j) * kK + 32 * s));
                const int4 v1 = __ldg(reinterpret_cast<const int4*>(qs + static_cast<std::int64_t>(j) * kK + 32 * s) + 1);
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                const std::uint8_t* b = w + static_cast<std::int64_t>(row + r) * kRowBytes +
                                        grp * kBlockBytes + it * 4 * kBlockBytes;
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

// an attention-like streamer: reads an int8 KV-shaped buffer with a 256-dim dot per key
__global__ void __launch_bounds__(128, 3) attn_like(const std::uint8_t* __restrict__ kv, int keys,
                                                    int splits, float* __restrict__ out) {
    const int per   = (keys + splits - 1) / splits;
    const int begin = blockIdx.x / 4 * per;         // 4 kv heads share the split
    const int end   = begin + per < keys ? begin + per : keys;
    float acc = 0.0f;
    for (int k = begin + (threadIdx.x >> 5); k < end; k += 4) {
        const std::uint8_t* row = kv + static_cast<std::int64_t>(k) * 256;
        for (int d = (threadIdx.x & 31) * 8; d < 256; d += 256) {
            const std::uint2 v = __ldg(reinterpret_cast<const std::uint2*>(row + d));
            acc += static_cast<float>(v.x & 0xFF) + static_cast<float>(v.y & 0xFF);
        }
    }
    if (acc == 1234.5f) { out[threadIdx.x] = acc; }
}

int main() {
    // the real per-tensor row counts, repeated to make a round's worth of launches
    const int row_mix[] = {17408, 17408, 6144, 6144, 1024, 1024, 2048, 14336, 34816, 5120};
    const int mix_n     = sizeof(row_mix) / sizeof(row_mix[0]);
    std::vector<std::uint8_t*> bufs(mix_n);
    std::size_t total = 0;
    for (int i = 0; i < mix_n; ++i) {
        const std::size_t bytes = static_cast<std::size_t>(row_mix[i]) * kRowBytes;
        CK(cudaMalloc(&bufs[i], bytes));
        CK(cudaMemset(bufs[i], 0x5A, bytes));
        total += bytes;
    }
    std::uint8_t* qs = nullptr;
    float2* ds = nullptr;
    float* sink = nullptr;
    std::uint8_t* kv = nullptr;
    float* aout = nullptr;
    CK(cudaMalloc(&qs, static_cast<std::size_t>(kT) * kK));
    CK(cudaMalloc(&ds, static_cast<std::size_t>(kT) * kSlices * sizeof(float2)));
    CK(cudaMalloc(&sink, kThreads * sizeof(float)));
    const int keys = 62400;                          // the 62.4k decode's key count
    CK(cudaMalloc(&kv, static_cast<std::size_t>(keys) * 256));
    CK(cudaMalloc(&aout, 128 * sizeof(float)));
    CK(cudaMemset(qs, 0x37, static_cast<std::size_t>(kT) * kK));
    CK(cudaMemset(ds, 0, static_cast<std::size_t>(kT) * kSlices * sizeof(float2)));
    CK(cudaMemset(kv, 0x21, static_cast<std::size_t>(keys) * 256));
    int dev = 0, sms = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const int cap = sms * 4;
    std::printf("device %d, %d SMs, %d tensors totalling %.0f MB, launch cap %d\n\n", dev, sms, mix_n,
                total / 1e6, cap);

    const int reps = 8;
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

    // H: many short launches over many separate tensors
    CK(cudaEventRecord(a));
    for (int rep = 0; rep < reps; ++rep) {
        for (int i = 0; i < mix_n; ++i) {
            const int groups = (row_mix[i] + kWarps * kRowsPerThread - 1) / (kWarps * kRowsPerThread);
            const int blocks = groups < cap ? groups : cap;
            vec_like<<<blocks, kThreads>>>(bufs[i], row_mix[i], sink, qs, ds);
        }
    }
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, a, b));
    std::printf("  H  %d short launches over %d tensors   %8.1f ms  %7.1f GB/s useful\n",
                reps * mix_n, mix_n, ms, total * reps / (ms * 1e-3) / 1e9);

    // I: the same, with an attention-like KV stream interleaved
    const int kv_splits = 130;                       // the real split count at 62.4k
    CK(cudaEventRecord(a));
    for (int rep = 0; rep < reps; ++rep) {
        for (int i = 0; i < mix_n; ++i) {
            const int groups = (row_mix[i] + kWarps * kRowsPerThread - 1) / (kWarps * kRowsPerThread);
            const int blocks = groups < cap ? groups : cap;
            vec_like<<<blocks, kThreads>>>(bufs[i], row_mix[i], sink, qs, ds);
            attn_like<<<kv_splits * 4, 128>>>(kv, keys, kv_splits, aout);
        }
    }
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b));
    const double kv_bytes = static_cast<double>(keys) * 256 * 2 * reps;   // K and V
    std::printf("  I  H plus a KV stream                %8.1f ms  %7.1f GB/s useful  (+%.2f GB of KV)\n",
                ms, total * reps / (ms * 1e-3) / 1e9, kv_bytes / 1e9);
    std::printf("     (if I is much slower than H per byte, the interleaving is the difference)\n");
    cudaFree(qs); cudaFree(ds); cudaFree(sink); cudaFree(kv); cudaFree(aout);
    for (auto* p : bufs) { cudaFree(p); }
    return 0;
}
