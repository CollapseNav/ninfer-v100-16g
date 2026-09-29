// Route-3 ceiling probe. Answers one question only: how fast is the CUTLASS sm70 FP16
// tensor-op GEMM that fp8_cutlass_sm70.cu already uses, on this model's real linear shapes,
// when the operands are plain fp16 in global memory with no dequant in the way?
//
// If this lands near 100 TFLOP/s, "dequantise once + reuse CUTLASS" beats the fused kernel's
// 36.7 TFLOP/s by ~2.5x and is worth the plumbing. If it lands near 40, the whole route is dead
// and the effort goes to the fused kernel instead.
//
// Built and run standalone inside the build container -- it does not touch build-sm70.
#include "cutlass/bfloat16.h"
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

// Byte-identical to the Gemm typedef in src/ops/linear/fp8/fp8_cutlass_sm70.cu.
using Gemm = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor, ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
        ElementComputeEpilogue>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

#define CK(x)                                                                        \
    do {                                                                             \
        cudaError_t e = (x);                                                         \
        if (e != cudaSuccess) {                                                      \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, \
                        __LINE__);                                                   \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

static void bench(int m, int n, int k, int reps, const char* tag) {
    ElementInput* a = nullptr;
    ElementInput* b = nullptr;
    ElementOutput* c = nullptr;
    CK(cudaMalloc(&a, static_cast<std::size_t>(m) * k * sizeof(ElementInput)));
    CK(cudaMalloc(&b, static_cast<std::size_t>(n) * k * sizeof(ElementInput)));
    CK(cudaMalloc(&c, static_cast<std::size_t>(m) * n * sizeof(ElementOutput)));
    CK(cudaMemset(a, 0x11, static_cast<std::size_t>(m) * k * sizeof(ElementInput)));
    CK(cudaMemset(b, 0x11, static_cast<std::size_t>(n) * k * sizeof(ElementInput)));

    const cutlass::gemm::GemmCoord shape(m, n, k);
    typename Gemm::Arguments args{shape, {a, k}, {b, k}, {c, n}, {c, n}, {1.0F, 0.0F}, 1};
    Gemm op;
    if (op.can_implement(args) != cutlass::Status::kSuccess) {
        std::printf("  %-28s can_implement FAILED\n", tag);
        return;
    }
    if (op.initialize(args, nullptr, nullptr) != cutlass::Status::kSuccess) {
        std::printf("  %-28s initialize FAILED\n", tag);
        return;
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

    const double flop = 2.0 * static_cast<double>(m) * n * k;
    const double tflops = flop / (ms * 1e-3) / 1e12;
    const double bytes = static_cast<double>(m) * k * 2 + static_cast<double>(n) * k * 2 +
                         static_cast<double>(m) * n * 2;
    const double gbs = bytes / (ms * 1e-3) / 1e9;
    std::printf("  %-28s %8.3f ms  %7.1f TFLOP/s  %6.1f%% of 125  |  %6.1f GB/s AI=%5.1f\n", tag,
                ms, tflops, tflops / 125.0 * 100.0, gbs, flop / bytes);

    CK(cudaFree(a));
    CK(cudaFree(b));
    CK(cudaFree(c));
}

int main(int argc, char** argv) {
    const int t = argc > 1 ? std::atoi(argv[1]) : 3412;
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s sm_%d%d SMs=%d sharedPerSM=%zu\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, prop.sharedMemPerMultiprocessor);
    std::printf("CUTLASS sm70 fp16 tensor-op GEMM, M=%d, plain fp16 operands:\n", t);
    bench(t, 17408, 5120, 20, "qkv/gate_up 17408x5120");
    bench(t, 5120, 17408, 20, "o/down 5120x17408");
    bench(t, 248320, 5120, 5, "lm_head 248320x5120");
    // Decode-shaped sanity check: M=1 collapses the GEMM to a GEMV, which CUTLASS does badly.
    bench(1, 17408, 5120, 50, "decode M=1 17408x5120");
    bench(16, 17408, 5120, 50, "mtp/verify M=16 17408x5120");
    return 0;
}
