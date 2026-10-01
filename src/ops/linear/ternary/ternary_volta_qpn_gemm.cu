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
    constexpr int kSplitk = 4;
    constexpr int kNacc   = 1;
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
    // SPLITK = 4 warps split K by whole groups, and every warp must get at least one.
    if ((k / TernaryVoltaQpnSchedule::kGroupK) % 4 != 0) { return false; }
    // Two 8-token tiles is the whole verify/lookup band this route targets.
    return t <= 2 * TernaryVoltaQpnSchedule::kRowsPerTile;
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
        if (tiles <= 1) { launch_shape<1, half>(x, w, out, out_row_stride, stream); }
        else            { launch_shape<2, half>(x, w, out, out_row_stride, stream); }
    } else {
        if (tiles <= 1) { launch_shape<1, __nv_bfloat16>(x, w, out, out_row_stride, stream); }
        else            { launch_shape<2, __nv_bfloat16>(x, w, out, out_row_stride, stream); }
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
