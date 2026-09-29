// Tile-shape sweep for the route-3 GEMM. The in-situ CUTLASS arm measures ~47 TFLOP/s against the
// 99 TFLOP/s the 128x128x32 tile reaches on a lone GEMM, and the difference is B-panel re-reads:
// B is re-read (M/BM) times, so BM is the only lever on the dominant traffic term.
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/half.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using ElementAccumulator     = float;
using ElementComputeEpilogue = float;
using ElementInput           = cutlass::half_t;
using ElementOutput          = cutlass::bfloat16_t;

#define CK(x)                                                                        \
    do {                                                                             \
        cudaError_t e = (x);                                                         \
        if (e != cudaSuccess) {                                                      \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, \
                        __LINE__);                                                   \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

template <int BM, int BN, int BK, int WBM, int WBN, int Stages>
using GemmT = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor, ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<BM, BN, BK>,
    cutlass::gemm::GemmShape<WBM, WBN, BK>, cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
        ElementComputeEpilogue>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages>;

// The device GEMM exposes the threadblock tile through its ThreadblockShape.
template <class Gemm>
struct GemmTileBM {
    static constexpr int value = Gemm::ThreadblockShape::kM;
};


template <class Gemm>
bool run(int m, int n, int k, int reps, const char* tag, const ElementInput* a,
         const ElementInput* b, ElementOutput* c) {
    const cutlass::gemm::GemmCoord shape(m, n, k);
    typename Gemm::Arguments args{shape, {a, k}, {b, k}, {c, n}, {c, n}, {1.0F, 0.0F}, 1};
    Gemm op;
    if (op.can_implement(args) != cutlass::Status::kSuccess) {
        std::printf("  %-34s can_implement FAILED\n", tag);
        return false;
    }
    if (op.initialize(args, nullptr, nullptr) != cutlass::Status::kSuccess) {
        std::printf("  %-34s initialize FAILED\n", tag);
        return false;
    }
    CK(cudaDeviceSynchronize());
    for (int i = 0; i < 2; ++i) { (void)op(nullptr); }
    CK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i = 0; i < reps; ++i) { (void)op(nullptr); }
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms = 0.0F;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= static_cast<float>(reps);
    const double flop = 2.0 * m * n * k;
    const double tflops = flop / (ms * 1e-3) / 1e12;
    // B traffic with an ideal L2 for A: B is re-read ceil(M/BM) times by construction.
    const double b_read = static_cast<double>(n) * k * 2 *
                          ((m + GemmTileBM<Gemm>::value - 1) / GemmTileBM<Gemm>::value);
    std::printf("  %-34s %8.3f ms  %6.1f TFLOP/s  %5.1f%%  B=%6.1f GB (%5.0f GB/s)\n", tag, ms,
                tflops, tflops / 125.0 * 100.0, b_read / 1e9, b_read / (ms * 1e-3) / 1e9);
    return true;
}

int main(int argc, char** argv) {
    const int t = argc > 1 ? std::atoi(argv[1]) : 3412;
    ElementInput* a = nullptr;
    ElementInput* b = nullptr;
    ElementOutput* c = nullptr;
    CK(cudaMalloc(&a, static_cast<std::size_t>(t) * 17408 * 2));
    CK(cudaMalloc(&b, 34816ull * 17408 * 2));
    CK(cudaMalloc(&c, static_cast<std::size_t>(t) * 34816 * 2));
    CK(cudaMemset(a, 0x11, static_cast<std::size_t>(t) * 17408 * 2));
    CK(cudaMemset(b, 0x11, 34816ull * 17408 * 2));

    // The two dominant families, at the real shapes and the real prompt length.
    struct Case { int n, k; const char* name; };
    const Case cases[] = {{34816, 5120, "gate_up 34816x5120"}, {5120, 17408, "down 5120x17408"}};
    for (const Case& cs : cases) {
        std::printf("\nM=%d  %s\n", t, cs.name);
        run<GemmT<128, 128, 32, 64, 64, 2>>(t, cs.n, cs.k, 10, "128x128x32 s2 (current)", a, b, c);
        run<GemmT<256, 128, 32, 64, 64, 2>>(t, cs.n, cs.k, 10, "256x128x32 s2", a, b, c);
        run<GemmT<128, 256, 32, 64, 64, 2>>(t, cs.n, cs.k, 10, "128x256x32 s2", a, b, c);
        run<GemmT<256, 256, 32, 64, 64, 2>>(t, cs.n, cs.k, 10, "256x256x32 s2", a, b, c);
        run<GemmT<128, 128, 64, 64, 64, 2>>(t, cs.n, cs.k, 10, "128x128x64 s2", a, b, c);
        run<GemmT<256, 128, 64, 64, 64, 2>>(t, cs.n, cs.k, 10, "256x128x64 s2", a, b, c);
    }
    return 0;
}
