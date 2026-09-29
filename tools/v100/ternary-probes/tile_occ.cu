// Occupancy sweep for the sm70 tensor-op GEMM that route 3 uses.
//
// Why this exists. Deeper software pipelining is not available: CUTLASS's sm70 tensor-op
// kernel::DefaultGemm is specialised for Stages == 2 only, and the threadblock kernel behind it is
// MmaPipelined, which asserts kStages == 2. So the only remaining way to get more loads in flight
// on Volta is more resident CTAs per SM. That is a function of the threadblock tile: the tile fixes
// the per-thread accumulator count (BM*BN/threads floats) and therefore the register footprint.
//
// Reports, per variant: threads, registers, static+dynamic shared, the occupancy
// cudaOccupancyMaxActiveBlocksPerMultiprocessor returns for the real kernel, and TFLOP/s plus the
// B-panel traffic rate at the model's two dominant shapes.
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/half.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

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

template <int BM, int BN, int BK, int WBM, int WBN>
using GemmT = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor, ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<BM, BN, BK>,
    cutlass::gemm::GemmShape<WBM, WBN, BK>, cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
        ElementComputeEpilogue>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

// The device GEMM exposes the kernel type, so the kernel function itself can be asked about
// registers, shared memory and occupancy -- no guessing from the tile. cutlass::Kernel is a
// __global__ function template, so it has to be taken by address, not named as a type.
template <class G>
void describe(const char* tag) {
    const void* kfn = reinterpret_cast<const void*>(&cutlass::Kernel<typename G::GemmKernel>);
    cudaFuncAttributes attr{};
    CK(cudaFuncGetAttributes(&attr, kfn));
    int blocks = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kfn, G::GemmKernel::kThreadCount, 0));
    const int kM = G::ThreadblockShape::kM;
    const int kN = G::ThreadblockShape::kN;
    std::printf("  %-30s thr=%4d regs=%3d smem=%6zu B=%d/SM  tile=%dx%d\n", tag,
                G::GemmKernel::kThreadCount, attr.numRegs, attr.sharedSizeBytes, blocks, kM, kN);
}

template <class G>
bool run(int m, int n, int k, int reps, const char* tag, const ElementInput* a,
         const ElementInput* b, ElementOutput* c) {
    const cutlass::gemm::GemmCoord shape(m, n, k);
    typename G::Arguments args{shape, {a, k}, {b, k}, {c, n}, {c, n}, {1.0F, 0.0F}, 1};
    G op;
    if (op.can_implement(args) != cutlass::Status::kSuccess) {
        std::printf("  %-30s can_implement FAILED\n", tag);
        return false;
    }
    if (op.initialize(args, nullptr, nullptr) != cutlass::Status::kSuccess) {
        std::printf("  %-30s initialize FAILED\n", tag);
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
    const double tflops = 2.0 * m * n * k / (ms * 1e-3) / 1e12;
    const double b_read =
        static_cast<double>(n) * k * 2 * ((m + G::ThreadblockShape::kM - 1) / G::ThreadblockShape::kM);
    std::printf("  %-30s %7.3f ms  %6.1f TFLOP/s  %5.1f%%  B=%6.1f GB (%5.0f GB/s)\n", tag, ms,
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

    std::printf("=== occupancy (kernel attributes, not estimates) ===\n");
    describe<GemmT<128, 128, 32, 64, 64>>("128x128x32 w64x64 (current)");
    describe<GemmT<64, 128, 32, 32, 64>>("64x128x32 w32x64");
    describe<GemmT<64, 128, 32, 64, 64>>("64x128x32 w64x64");
    describe<GemmT<128, 64, 32, 64, 32>>("128x64x32 w64x32");
    describe<GemmT<64, 64, 32, 32, 32>>("64x64x32 w32x32");
    describe<GemmT<128, 128, 32, 32, 64>>("128x128x32 w32x64");
    describe<GemmT<256, 128, 32, 64, 64>>("256x128x32 w64x64");
    describe<GemmT<128, 256, 32, 64, 64>>("128x256x32 w64x64");

    struct Case { int n, k; const char* name; };
    const Case cases[] = {{34816, 5120, "gate_up 34816x5120"}, {5120, 17408, "down 5120x17408"}};
    for (const Case& cs : cases) {
        std::printf("\nM=%d  %s\n", t, cs.name);
        run<GemmT<128, 128, 32, 64, 64>>(t, cs.n, cs.k, 10, "128x128x32 w64x64 (current)", a, b, c);
        run<GemmT<64, 128, 32, 32, 64>>(t, cs.n, cs.k, 10, "64x128x32 w32x64", a, b, c);
        run<GemmT<64, 128, 32, 64, 64>>(t, cs.n, cs.k, 10, "64x128x32 w64x64", a, b, c);
        run<GemmT<128, 64, 32, 64, 32>>(t, cs.n, cs.k, 10, "128x64x32 w64x32", a, b, c);
        run<GemmT<64, 64, 32, 32, 32>>(t, cs.n, cs.k, 10, "64x64x32 w32x32", a, b, c);
        run<GemmT<128, 128, 32, 32, 64>>(t, cs.n, cs.k, 10, "128x128x32 w32x64", a, b, c);
        run<GemmT<128, 128, 64, 64, 64>>(t, cs.n, cs.k, 10, "128x128x64 w64x64", a, b, c);
    }
    return 0;
}
