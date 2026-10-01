// PQ2 Volta tensor-core (QPN) launcher: shape gate, instantiation table and the output policy.
// Fifth sibling of nvfp4/fp8/w8/q4_volta_qpn_gemm; see ternary_volta_qpn_gemm.cuh for the design.
#include "ops/linear/ternary/ternary_volta_qpn_gemm.h"

#include "core/device.h"
#include "ops/linear/ternary/ternary_dispatch.h"
#include "ops/linear/ternary/ternary_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

#include <cstdio>

#include <cstdlib>
#include <string>
#include <unordered_set>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {

// SPLITK. The sibling NVFP4 kernel calls 8 the production floor, and the reason is parallelism
// rather than the K-split itself: one CTA covers only 32 output rows, so the grid is n/32 (160 CTAs
// for n = 5120) and at SPLITK = 4 that is 8 warps per SM -- nowhere near enough to hide a weight
// stream's latency. Every width in this model has a group count divisible by 8 (40, 48, 80, 136).
constexpr int kTernaryQpnSplitk = 8;

// [n, t] with n contiguous per token, bf16, honouring the caller's row stride.
struct TernaryBf16Output {
    __nv_bfloat16* data;
    std::int32_t row_stride;

    __device__ __forceinline__ void store(std::int32_t col, std::int32_t token, float value) const {
        data[static_cast<std::int64_t>(token) * row_stride + col] = __float2bfloat16_rn(value);
    }
};

template <int kTiles, class Activation>
void launch_shape(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                   cudaStream_t stream) {
    using S = TernaryVoltaQpnSchedule;
    constexpr int kSplitk = kTernaryQpnSplitk;
    // One accumulator per k-slice of a 16-k unit would break the mma RAW chain (the sibling kernels'
    // NACC knob), but four accumulators per tile is 32 registers per live tile and kTiles goes to 4;
    // with more than one tile the tiles are already independent, which is the same ILP for free.
    constexpr int kNacc   = kTiles == 1 ? 4 : 1;
    const int n = w.n;
    const unsigned grid = static_cast<unsigned>((n + S::kColsPerCta - 1) / S::kColsPerCta);
    ternary_volta_qpn_gemm_kernel<kTiles, kSplitk, kNacc, TernaryBf16Output, Activation>
        <<<grid, kSplitk * 32, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
            static_cast<const Activation*>(x.data), n, w.k, x.ne[1],
            TernaryBf16Output{static_cast<__nv_bfloat16*>(out.data), out_row_stride});
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

#endif // NINFER_VOLTA_BUILD

#ifdef NINFER_VOLTA_BUILD
namespace {

__global__ void pq2_prepack_tile_kernel(const std::uint8_t* __restrict__ src_codes,
                                        const std::uint16_t* __restrict__ src_scales,
                                        std::uint8_t* __restrict__ dst_codes,
                                        std::uint16_t* __restrict__ dst_scales, int groups) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int tile = static_cast<int>(blockIdx.x);
    const int qp   = (lane >> 2) & 3;
    const int r    = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);
    const int row  = tile * 32 + qp * 8 + r;
    const int stride = static_cast<int>(blockDim.x) >> 5;
    for (int g = static_cast<int>(threadIdx.x) >> 5; g < groups; g += stride) {
        const std::int64_t packed = (static_cast<std::int64_t>(tile) * groups + g) * 32 + lane;
        const std::uint8_t* src = src_codes + (static_cast<std::int64_t>(row) * groups + g) * 32;
        std::uint8_t* dst       = dst_codes + packed * 32;
#pragma unroll
        for (int b = 0; b < 32; ++b) { dst[b] = src[b]; }
        dst_scales[packed] = src_scales[static_cast<std::int64_t>(row) * groups + g];
    }
}

std::unordered_set<const void*>& prepacked_weights() {
    static std::unordered_set<const void*> set;
    return set;
}

} // namespace
#endif

bool ternary_qpn_is_prepacked(const void* qdata) noexcept {
    return qdata != nullptr && prepacked_weights().count(qdata) != 0;
}

void ternary_prepack_qpn(Weight& w, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    const int groups = w.k / TernaryVoltaQpnSchedule::kGroupK;
    const std::size_t code_bytes  = static_cast<std::size_t>(w.n) * groups * 32;
    const std::size_t scale_bytes = static_cast<std::size_t>(w.n) * groups * sizeof(std::uint16_t);
    void* scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, code_bytes + scale_bytes));
    auto* packed_codes  = static_cast<std::uint8_t*>(scratch);
    auto* packed_scales = reinterpret_cast<std::uint16_t*>(packed_codes + code_bytes);
    pq2_prepack_tile_kernel<<<static_cast<unsigned>(w.n / 32), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata),
        reinterpret_cast<const std::uint16_t*>(w.scales), packed_codes, packed_scales, groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(w.qdata), packed_codes, code_bytes,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(w.scales), packed_scales, scale_bytes,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(scratch));
    prepacked_weights().insert(w.qdata);
#else
    (void)w;
    (void)stream;
#endif
}

bool ternary_qpn_enabled() noexcept {
    static const bool value = [] {
        const char* env = std::getenv("NINFER_TERNARY_QPN");
        return env != nullptr && std::string(env) != "0";
    }();
    return value;
}

bool ternary_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
#ifdef NINFER_VOLTA_BUILD
    if (n <= 0 || k <= 0 || t <= 0) { return false; }
    if ((k % TernaryVoltaQpnSchedule::kGroupK) != 0) { return false; }
    // SPLITK warps split K by whole groups, and every warp must get at least one.
    if ((k / TernaryVoltaQpnSchedule::kGroupK) % kTernaryQpnSplitk != 0) { return false; }
    // T = 1 is deliberately not on this route: the kernel's A tile is eight tokens tall whatever T
    // is, so one token runs the whole mma schedule with seven eighths of M dead and one CTA per 32
    // output rows. Measured 32.6 t/s against the block GEMV's 49.8 on the parked configuration.
    // The verify band above it is still a LOSS at this commit (T = 2 by 47%, T = 3 by 21%); the
    // band gate and the measured crossover table land in the next commit.
    if (t < 2) { return false; }
    return t <= 4 * TernaryVoltaQpnSchedule::kRowsPerTile;
#else
    (void)n;
    (void)k;
    (void)t;
    return false;
#endif
}

void launch_ternary_volta_qpn(const Tensor& x, const Weight& w, Tensor& out,
                              std::int32_t out_row_stride, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    const int tiles = (x.ne[1] + TernaryVoltaQpnSchedule::kRowsPerTile - 1) /
                      TernaryVoltaQpnSchedule::kRowsPerTile;
    if (x.dtype == DType::FP16) {
        if (tiles <= 1)      { launch_shape<1, half>(x, w, out, out_row_stride, stream); }
        else if (tiles == 2) { launch_shape<2, half>(x, w, out, out_row_stride, stream); }
        else if (tiles == 3) { launch_shape<3, half>(x, w, out, out_row_stride, stream); }
        else                 { launch_shape<4, half>(x, w, out, out_row_stride, stream); }
    } else {
        if (tiles <= 1)      { launch_shape<1, __nv_bfloat16>(x, w, out, out_row_stride, stream); }
        else if (tiles == 2) { launch_shape<2, __nv_bfloat16>(x, w, out, out_row_stride, stream); }
        else if (tiles == 3) { launch_shape<3, __nv_bfloat16>(x, w, out, out_row_stride, stream); }
        else                 { launch_shape<4, __nv_bfloat16>(x, w, out, out_row_stride, stream); }
    }
#else
    (void)x;
    (void)w;
    (void)out;
    (void)out_row_stride;
    (void)stream;
#endif
}

} // namespace ninfer::ops::detail
