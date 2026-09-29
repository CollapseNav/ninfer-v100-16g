// Decode is 41.6 t/s against a 125 t/s HBM floor, i.e. 298 GB/s of 900. The T=1 GEMV keeps one
// warp per output row and has each lane load ONE code byte per group, so a warp issues one 32-byte
// transaction per group walk step. The kernel is measured to be memory-level-parallelism bound (the
// unroll factor swings throughput 39% while deleting instructions swings it 0%), which is what a
// 32-byte-per-instruction stream looks like.
//
// This measures the read pattern alone, on this model's real PQ2 geometry, sweeping the bytes each
// lane takes per step. If wide loads do not lift the achieved bandwidth here, they will not lift the
// GEMV either, and the effort is better spent elsewhere.
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

// W bytes per lane per step; a warp therefore consumes 32*W bytes = W groups of 32 code bytes.
template <int W>
__global__ __launch_bounds__(256)
void read_row_pattern(const std::uint8_t* __restrict__ codes, int rows, int groups_per_row,
                      unsigned* __restrict__ sink) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }
    const std::uint8_t* row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * 32;
    unsigned acc = 0;
    int g = 0;
    for (; g + W <= groups_per_row; g += W) {
        const std::uint8_t* p = row + g * 32 + lane * W;
        if constexpr (W == 1) {
            acc += p[0];
        } else if constexpr (W == 2) {
            acc += *reinterpret_cast<const std::uint16_t*>(p);
        } else if constexpr (W == 4) {
            acc += *reinterpret_cast<const std::uint32_t*>(p);
        } else if constexpr (W == 8) {
            const uint2 v = *reinterpret_cast<const uint2*>(p);
            acc += v.x + v.y;
        } else {
            const uint4 v = *reinterpret_cast<const uint4*>(p);
            acc += v.x + v.y + v.z + v.w;
        }
    }
    if (acc == 0xdeadbeefu) { sink[0] = acc; }
}

// The same volume read as one flat array, ignoring the row structure. This is the ceiling the
// per-row pattern is being compared against.
__global__ __launch_bounds__(256)
void read_flat(const uint4* __restrict__ codes, std::int64_t vecs,
               unsigned* __restrict__ sink) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    unsigned acc = 0;
    for (std::int64_t j = i; j < vecs; j += stride) {
        const uint4 v = codes[j];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 0xdeadbeefu) { sink[0] = acc; }
}

// The row pattern WITH the arithmetic: 4 FMAs per code byte, as the real GEMV does.
template <int W>
__global__ __launch_bounds__(256)
void read_row_pattern_fma(const std::uint8_t* __restrict__ codes,
                          const float* __restrict__ act, int rows, int groups_per_row,
                          unsigned* __restrict__ sink) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }
    const std::uint8_t* row = codes + static_cast<std::int64_t>(warp) * groups_per_row * 32;
    float acc = 0.0F;
    int g = 0;
    for (; g + W <= groups_per_row; g += W) {
        const std::uint8_t* p = row + g * 32 + lane * W;
        float dot = 0.0F;
#pragma unroll
        for (int b = 0; b < W; ++b) {
            const unsigned byte = p[b];
            // The activation is indexed by the same walk the real kernel uses; it is L1-resident
            // because every warp on the row reads the same window.
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float w = static_cast<float>(static_cast<int>((byte >> (2 * c)) & 3u) - 1);
                dot = fmaf(w, act[(g * 32 + lane * W + b) * 4 + c], dot);
            }
        }
        acc += dot;
    }
    if (acc == 0xdeadbeefu) { sink[0] = 1u; }
}

static void bench(const char* tag, double bytes, int iters, void (*launch)(int)) {
    (void)tag;
    (void)bytes;
    (void)iters;
    (void)launch;
}

int main(int argc, char** argv) {
    (void)bench;
    // This model's widest GEMV is n=34816, k=5120 -> 40 groups per row, 1280 B row stride. One such
    // matrix is only 44.6 MB, which is a 50 us kernel -- launch overhead would dominate the
    // measurement -- so the row count is replicated to the model's real code-plane volume (~713 MB).
    const int rows = 34816 * 16;
    const int groups = 40;
    const std::int64_t total = static_cast<std::int64_t>(rows) * groups * 32;
    std::uint8_t* codes = nullptr;
    float* act = nullptr;
    unsigned* sink = nullptr;
    CK(cudaMalloc(&codes, total));
    CK(cudaMemset(codes, 0x55, total));
    CK(cudaMalloc(&act, static_cast<std::size_t>(groups) * 32 * 4 * sizeof(float)));
    CK(cudaMemset(act, 0, static_cast<std::size_t>(groups) * 32 * 4 * sizeof(float)));
    CK(cudaMalloc(&sink, 4));

    const int reps = argc > 1 ? std::atoi(argv[1]) : 120;
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));

    auto timeit = [&](const char* tag, double bytes_per_rep, auto&& fn) {
        for (int i = 0; i < 3; ++i) { fn(); }
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(t0));
        for (int i = 0; i < reps; ++i) { fn(); }
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms = 0.0F;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= static_cast<float>(reps);
        const double gbs = bytes_per_rep / (ms * 1e-3) / 1e9;
        std::printf("  %-34s %8.3f ms  %7.1f GB/s\n", tag, ms, gbs);
    };

    const dim3 grid(rows / 8, 1, 1);
    std::printf("rows=%d groups/row=%d volume=%.1f MB  reps=%d\n", rows, groups, total / 1e6, reps);
    // Only the part the template actually walks is counted: the loop stops at the last whole step.
    auto walked = [&](int W) {
        const int steps = groups / W;
        return static_cast<double>(rows) * steps * 32.0 * W;
    };

    timeit("flat uint4 (ceiling)", static_cast<double>(total), [&] {
        read_flat<<<2048, 256>>>(reinterpret_cast<const uint4*>(codes), total / 16, sink);
    });
    timeit("row pattern W=1  (today)", walked(1),
           [&] { read_row_pattern<1><<<grid, 256>>>(codes, rows, groups, sink); });
    timeit("row pattern W=2", walked(2),
           [&] { read_row_pattern<2><<<grid, 256>>>(codes, rows, groups, sink); });
    timeit("row pattern W=4", walked(4),
           [&] { read_row_pattern<4><<<grid, 256>>>(codes, rows, groups, sink); });
    timeit("row pattern W=8", walked(8),
           [&] { read_row_pattern<8><<<grid, 256>>>(codes, rows, groups, sink); });
    timeit("row pattern W=16", walked(16),
           [&] { read_row_pattern<16><<<grid, 256>>>(codes, rows, groups, sink); });
    std::printf("  (W=16 skips the last 8 of 40 groups; the byte count above is what it walks)\n\n");
    timeit("row W=1  + 4 FMA/byte", walked(1),
           [&] { read_row_pattern_fma<1><<<grid, 256>>>(codes, act, rows, groups, sink); });
    timeit("row W=8  + 4 FMA/byte", walked(8),
           [&] { read_row_pattern_fma<8><<<grid, 256>>>(codes, act, rows, groups, sink); });
    timeit("row W=16 + 4 FMA/byte", walked(16),
           [&] { read_row_pattern_fma<16><<<grid, 256>>>(codes, act, rows, groups, sink); });
    CK(cudaDeviceSynchronize());
    return 0;
}
