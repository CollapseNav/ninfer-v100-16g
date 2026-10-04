// vec_probe5.cu -- the activation operand is the bottleneck (probe4: adding it costs 10.6x, 769 -> 72
// GB/s). Its cost per weight byte is (T x K activation bytes) / (rows sharing one activation read), so
// the lever is how many rows each activation load is amortised over. The real kernel has kRows=2 and
// raising it lost because of registers; here, with no other pressure, the ratio can be tested directly.
//
//   D  decode, activations in registers, kRows=2   (the 769 GB/s baseline)
//   E  decode + activation loads, kRows=2          (the 72 GB/s baseline)
//   F  decode + activation loads, kRows=4
//   G  decode + activation loads, kRows=8

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kIters   = 5;
constexpr int kBlockBytes = 74;
constexpr int kRowBytes   = (5120 / 256) * 74;
constexpr int kK = 5120;
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
__device__ __forceinline__ std::uint32_t ksigns(std::uint32_t v) {
    return v ^ ((__popc(v) & 1) << 7);
}

// the decode of one sub-block: 6 weight loads -> 8 int32 operands
__device__ __forceinline__ void decode_block(const std::uint8_t* b, int u, const uint2* grid, int w0[8]) {
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

__device__ __forceinline__ void load_act(const std::uint8_t* qs, const float2* ds, int s, int a[kT][8],
                                         float d[kT]) {
#pragma unroll
    for (int j = 0; j < kT; ++j) {
        d[j] = __ldg(&ds[static_cast<std::int64_t>(j) * kSlices + s]).x;
        const int4 v0 = __ldg(reinterpret_cast<const int4*>(qs + static_cast<std::int64_t>(j) * kK + 32 * s));
        const int4 v1 = __ldg(reinterpret_cast<const int4*>(qs + static_cast<std::int64_t>(j) * kK + 32 * s) + 1);
        a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
        a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
    }
}

// ROWS = rows carried per thread, and the activation load happens once per ROWS rows.
template <int ROWS>
__global__ void __launch_bounds__(kThreads, 4) probe_act(
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
    for (int row = (blockIdx.x * kWarps + warp) * ROWS; row < rows;
         row += gridDim.x * kWarps * ROWS) {
#pragma unroll
        for (int it = 0; it < kIters; ++it) {
            const int s = lane + 32 * it;
            int a[kT][8];
            float d[kT];
            load_act(qs, ds, s, a, d);
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const std::uint8_t* b = w + static_cast<std::int64_t>(row + r) * kRowBytes +
                                        grp * kBlockBytes + it * 4 * kBlockBytes;
                int w0[8];
                decode_block(b, u, grid, w0);
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

// D: the decode with the activations held in registers (no activation traffic).
__global__ void __launch_bounds__(kThreads, 4) probe_noact(
    const std::uint8_t* __restrict__ w, int rows, float* __restrict__ sink,
    const std::uint8_t* __restrict__ qs, const float2* __restrict__ ds) {
    (void)qs; (void)ds;
    __shared__ __align__(16) std::uint32_t table[1024];
    for (int i = threadIdx.x; i < 1024; i += kThreads) {
        table[i] = __ldg(reinterpret_cast<const std::uint32_t*>(w) + i);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int u = lane & 7, grp = lane >> 3;
    const uint2* grid = reinterpret_cast<const uint2*>(table);
    int a[kT][8];
#pragma unroll
    for (int j = 0; j < kT; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) { a[j][i] = (lane * 31 + j * 7 + i * 3) & 0xFF; }
    float acc = 0.0f;
    for (int row = (blockIdx.x * kWarps + warp) * 2; row < rows; row += gridDim.x * kWarps * 2) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const std::uint8_t* base = w + static_cast<std::int64_t>(row + r) * kRowBytes;
#pragma unroll
            for (int it = 0; it < kIters; ++it) {
                const std::uint8_t* b = base + grp * kBlockBytes + it * 4 * kBlockBytes;
                int w0[8];
                decode_block(b, u, grid, w0);
#pragma unroll
                for (int j = 0; j < kT; ++j) {
                    int s0 = 0, s1 = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) { s0 = __dp4a(w0[i], a[j][i], s0); }
#pragma unroll
                    for (int i = 0; i < 4; ++i) { s1 = __dp4a(w0[4 + i], a[j][4 + i], s1); }
                    acc += static_cast<float>(s0 + s1);
                }
            }
        }
    }
    sink[threadIdx.x] = acc;
}

template <typename K>
static int run(const char* name, K kernel, const std::uint8_t* w, int rows, float* sink, int blocks,
               int reps, const std::uint8_t* qs, const float2* ds) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    kernel<<<blocks, kThreads>>>(w, rows, sink, qs, ds);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) { kernel<<<blocks, kThreads>>>(w, rows, sink, qs, ds); }
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, a, b));
    const double bytes = static_cast<double>(rows) * kRowBytes * reps;
    std::printf("  %-38s %8.1f ms  %7.1f GB/s useful\n", name, ms, bytes / (ms * 1e-3) / 1e9);
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
    std::printf("device %d, %d SMs, %.0f MB random weights, T=%d\n\n", dev, sms, bytes / 1e6, kT);
    // grid sized the way the shipped launch does: min(groups, resident * sm_count)
    auto grid_for = [&](int rows_per_thread) {
        const int groups = (rows + kWarps * rows_per_thread - 1) / (kWarps * rows_per_thread);
        return groups < sms * 4 ? groups : sms * 4;
    };
    if (run("D decode, activations in regs", probe_noact, w, rows, sink, grid_for(2), 20, qs, ds)) return 1;
    if (run("E decode + act loads, 2 rows", probe_act<2>, w, rows, sink, grid_for(2), 20, qs, ds)) return 1;
    if (run("F decode + act loads, 4 rows", probe_act<4>, w, rows, sink, grid_for(4), 20, qs, ds)) return 1;
    if (run("G decode + act loads, 8 rows", probe_act<8>, w, rows, sink, grid_for(8), 20, qs, ds)) return 1;
    cudaFree(w); cudaFree(sink); cudaFree(qs); cudaFree(ds);
    return 0;
}
