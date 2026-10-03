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

// NACC is the sibling kernels' generation-2 knob: it round-robins the four mma of one 16-k unit into
// independent accumulators so the RAW chain on the accumulator is not four deep. The built-in
// default is 4 at kTiles = 1 (the decode and verify tiles have no other ILP) and 1 elsewhere, with
// **2 at kTiles = 2** since that change -- T = 9..16, the lookup verify where the route is worth
// +90% and the mma count doubles with the tile.
//
// Measured, same batch, lookup10 K = 7: NACC 1 -> 235.5, NACC 2 -> 237.5 (+0.85%), NACC 4 -> 227.6
// (-3.4%), output text identical at all three. Continuous check at --context 16 --stride 8, which is
// the only window plan that lands every forward on kTiles = 2, both arms on the QPN route, 8,675
// scored tokens: mean_nll 4.165766 (NACC 1) against 4.166126 (NACC 2), i.e. +0.036% PPL. That is a
// real numerics cost -- 2.5x the QPN band's own deviation from SIMT (+0.0143%) and 1.5x the fp16
// operand margin this tree accepted for its prefill routes (+0.0245%) -- bought for +0.85% on one
// path, so `=1` is the escape hatch for anyone who wants the tighter numbers.
//
// The override now reaches EVERY kTiles instead of only kTiles = 2. It did not before, which made
// the knob silently inert on the T = 1..8 band (kTiles = 1 sits at NACC 4 without ever having been
// swept) and on T >= 17 -- the same class of dead default as the `NINFER_TERNARY_QPN` gate that
// round 6 of docs/decode-round-2026-10-01.md found. Unset still means "every kTiles at its built-in
// default", so nothing shipped moves.
int qpn_nacc_override() {
    static const int value = [] {
        const char* env = std::getenv("NINFER_TERNARY_QPN_NACC");
        if (env == nullptr) { return 0; }
        const int parsed = std::atoi(env);
        return (parsed == 1 || parsed == 2 || parsed == 4) ? parsed : 0;
    }();
    return value;
}

// The built-in default per kTiles, and the override applied on top. The kTiles = 4 shape cannot
// carry four accumulators per tile (32 registers per live tile, four tiles), so the override is
// refused above kTiles = 2 rather than spilling.
//
// kTiles = 1 defaults to 1, NOT 4. The 4 was inherited from the sibling kernels' note "the decode
// and verify tiles have no other ILP" and had never been swept on this route, and it is badly wrong:
// MEASURED (round 12), same batch, interleaved, two repetitions, with NINFER_TERNARY_QPN_MIN_T=1 so
// the whole T = 1 step runs QPN and the weight path is isolated (A tile 8 rows tall, 7 dead, no
// activation slope):
//
//   NACC      T = 1 whole step     MTP K = 1 (QPN verify only)
//   4 (old)   31.2 / 31.2 t/s      45.6 / 45.6
//   2         38.0 / 38.0          54.6 / 54.6     (+21.8% / +19.7%)
//   1         38.0 / 38.0          54.6 / 54.6     (tied with 2)
//
// SIMT control in the same batch: 55.2/55.3 at T = 1, 64.4 at MTP K = 1.
//
// IT ALSO MOVED THE LOOKUP PATH, which is how wrong the old default was outside the MTP band. The
// sweep's lookup10 arm read 235.0/235.2 t/s with the old default and 254.2/254.5 with this one, in
// two adjacent batches whose T = 1 SIMT control was identical (55.2/55.3 both) -- i.e. +8.2% causal,
// not drift. That was surprising, because lookup10's verify is T = 8 and the band was thought to sit
// at kTiles = 2; a trace settles it -- ninfer-perplexity and the CLI both run the verify at T = K + 1
// = 8, and tiles = ceil(8/8) = 1, so the lookup verify has ALWAYS been on this kTiles = 1 arm and the
// unswept NACC = 4 was holding it back. SHIPPED DEFAULT, same-batch smoke: lookup10 253.9/254.5,
// lookup16 265.0/265.4, against 237.4/237.6 and 249.8/249.8 before the change, with T = 1, MTP K = 1
// and MTP K = 3 unmoved to the digit and the md5 against /root/wt/base.out IDENTICAL.
//
// Two and one tying says the four-deep chain was not the binding cost at four accumulators -- the
// extra registers are. Past kTiles = 2 the tiles are independent, which is the same ILP for free,
// so 1 stays.
constexpr int qpn_nacc_for(int kTiles, int override_value) {
    if (override_value != 0 && kTiles <= 2) { return override_value; }
    return kTiles == 2 ? 2 : 1;
}

// NINFER_TERNARY_QPN_MINBLOCKS caps the QPN kernel's registers through __launch_bounds__'s second
// argument. The built-in value is 32/SPLITK = 4 for kTiles = 1, i.e. 50% occupancy of this SM, and it
// had never been swept -- the same lever that gave the dedicated T = 1 SIMT GEMV +4.8%.
//
// SPLITK is deliberately NOT swept alongside it. It has to divide the group count, and every width in
// this model has groups in {40, 48, 80, 136}, whose only common divisors are 1, 2, 4 and 8; 8 is the
// shipped value and 4 is the only alternative, which the constant's own comment already rejects
// ("one CTA covers only 32 output rows, so the grid is n/32 ... at SPLITK = 4 that is 8 warps per SM
// -- nowhere near enough to hide a weight stream's latency"). Recorded so it is not re-proposed.
int qpn_minblocks_override() {
    static const int value = [] {
        const char* env = std::getenv("NINFER_TERNARY_QPN_MINBLOCKS");
        if (env == nullptr) { return 0; }
        const int parsed = std::atoi(env);
        return (parsed == 4 || parsed == 6 || parsed == 8) ? parsed : 0;
    }();
    return value;
}

template <int kTiles, class Activation, int kNaccOverride = 0, int kMinBlocksOverride = 0>
void launch_shape(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                   cudaStream_t stream) {
    using S = TernaryVoltaQpnSchedule;
    constexpr int kSplitk = kTernaryQpnSplitk;
    // One accumulator per k-slice of a 16-k unit would break the mma RAW chain (the sibling kernels'
    // NACC knob), but four accumulators per tile is 32 registers per live tile and kTiles goes to 4;
    // with more than one tile the tiles are already independent, which is the same ILP for free.
    constexpr int kNacc = qpn_nacc_for(kTiles, kNaccOverride);
    const int n = w.n;
    const unsigned grid = static_cast<unsigned>((n + S::kColsPerCta - 1) / S::kColsPerCta);
    ternary_volta_qpn_gemm_kernel<kTiles, kSplitk, kNacc, TernaryBf16Output, Activation,
                                  kMinBlocksOverride>
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
    // Unset means enabled wherever the shape and the token count qualify; only "0" disables. This is
    // what the header, the task that commissioned the route, and the docs all assume -- the port
    // shipped the opposite test (`env != nullptr && env != "0"`), i.e. the route was OFF unless a
    // value was passed, and every measurement in this tree happened to pass one explicitly (0 or 1),
    // so the A/B was valid while the default was dead. Found by re-running the lookup arm with no
    // NINFER_TERNARY_QPN at all: 123.8 t/s (SIMT) against 235.1 with the route on.
    static const bool value = [] {
        const char* env = std::getenv("NINFER_TERNARY_QPN");
        return env == nullptr || std::string(env) != "0";
    }();
    return value;
}

bool ternary_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
#ifdef NINFER_VOLTA_BUILD
    if (n <= 0 || k <= 0 || t <= 0) { return false; }
    if ((k % TernaryVoltaQpnSchedule::kGroupK) != 0) { return false; }
    // SPLITK warps split K by whole groups, and every warp must get at least one.
    if ((k / TernaryVoltaQpnSchedule::kGroupK) % kTernaryQpnSplitk != 0) { return false; }
    // The band. The kernel reads the whole weight once per forward whatever T is, and the tensor
    // cores then absorb the extra tokens almost free -- so its cost is flat in T, while the SIMT
    // tile kernel's grows by ~a third of a T=1 step per verified token. Measured round times on
    // real_task (ms, NINFER_TERNARY_QPN=0 against =1, same batch, 128 tokens each):
    //
    //   T = 1    2     3     4     5     6     8     16 (lookup)
    //   19.5   26.6  33.9  41.2  48.6  58.0  74.1   98.1   SIMT tile
    //   19.4   39.2  41.1  43.0  44.9  47.5  52.0   46.8   QPN
    //
    // i.e. the two lines cross at T ~ 4.3. Below 6 the route loses (T = 2 by 47%, T = 3 by 21%),
    // which is the whole MTP band a real task uses -- K = 1 and K = 2 -- and 6 is the first width
    // where the win clears the +/-4% batch drift of this workload (18% at T = 6, 30% at T = 8, and
    // 2.1x at the T = 16 lookup verify). T = 1 is far worse than all of them: the A tile is eight
    // tokens tall whatever T is, so one token runs the schedule with seven eighths of M dead.
    //
    // NINFER_TERNARY_QPN_MIN_T moves the lower edge so that sweep can be re-run on a later build
    // instead of trusted. RE-MEASURED TWICE on the current build (rounds 10 and 12 of
    // docs/decode-round-2026-10-01.md), same batch, interleaved, two repetitions, round = acceptance
    // / decode_speed:
    //
    //   round 10, before the kTiles = 1 NACC fix:
    //     arm      K=1 (T=2) round   K=2 (T=3)   K=3 (T=4)
    //     min_t=6      25.62            33.67       40.97     <- then shipped
    //     min_t=4      25.62            33.73       43.49     (+6.2% at T=4)
    //     min_t=2      36.18  (+41%)    39.92       43.13
    //
    //   round 12, after it:
    //     min_t=6      25.62            33.73       41.01
    //     min_t=4      25.62            33.73       37.37     (-8.9% at T=4)
    //     min_t=3      25.62            33.93       37.37
    //     min_t=2      30.40  (+18.7%)  34.06       37.74
    //
    // So the crossover moved from T ~ 4.3 to T ~ 3.5. THE EDGE STAYS AT 6 ANYWAY, and the reason is
    // not the round: the T = 4 window costs +0.097% PPL against SIMT, four times the fp16 prefill
    // operand margin this tree accepted (+0.0245%) and 2.7x the NACC = 2 trade (+0.036%). And what
    // it buys is only that MTP K = 3 and K = 4 stop being so much worse than K = 1 -- they remain
    // worse. Per-token latency (round / acceptance) after the kTiles = 1 NACC fix:
    //
    //   K = 1   25.64 / 1.65 = 15.54 ms      K = 3   37.37 / 2.19 = 17.06 ms
    //   K = 2   33.79 / 1.85 = 18.27 ms      K = 4   41.10 / 2.40 = 17.13 ms
    //
    // K = 1's round does not move at all with the edge, so the change cannot improve the best
    // setting on this workload -- it pays a real numerical cost to make a losing setting less
    // losing. `NINFER_TERNARY_QPN_MIN_T=4` is left as the one-line escape hatch for a workload whose
    // acceptance curve makes K >= 3 worth running, and it is a trade, recorded as one.
    //
    // Continuous checks (round 12), both against SIMT with the same kTiles = 1 NACC default:
    //   forward width 4 (--context 5 --stride 4)  +0.097%   PPL  -> this is the T the edge moves
    //   forward width 7 (--context 8 --stride 4)  +0.0043%  PPL  -> kTiles = 1, below the tree's
    //                                                              accepted fp16 margin (+0.0245%)
    // ninfer-perplexity scores at a forward width of context - 1 -- traced: --context 4 runs
    // ternary_pq2_gemv_tile_kernel<int=3,...> and no QPN kernel at all -- which is what makes
    // --context 5 the T = 4 window.
    static const std::int32_t min_t = [] {
        const char* env    = std::getenv("NINFER_TERNARY_QPN_MIN_T");
        const int parsed   = env == nullptr ? 0 : std::atoi(env);
        return parsed >= 1 ? parsed : 6;
    }();
    if (t < min_t) { return false; }
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
    const int nacc_env = qpn_nacc_override();
    const int minb_env = qpn_minblocks_override();
    if (x.dtype == DType::FP16) {
        if (tiles <= 1) {
            // Built-in default here is NACC = 1 and min-blocks = 4; both overrides reach it now.
            const int nacc = qpn_nacc_for(1, nacc_env);
            if (nacc == 1 && minb_env == 6) {
                launch_shape<1, half, 1, 6>(x, w, out, out_row_stride, stream);
            } else if (nacc == 1 && minb_env == 8) {
                launch_shape<1, half, 1, 8>(x, w, out, out_row_stride, stream);
            } else if (nacc == 1) {
                launch_shape<1, half, 1>(x, w, out, out_row_stride, stream);
            } else if (nacc == 2) {
                launch_shape<1, half, 2>(x, w, out, out_row_stride, stream);
            } else {
                launch_shape<1, half>(x, w, out, out_row_stride, stream);
            }
        } else if (tiles == 2) {
            const int nacc = qpn_nacc_for(2, nacc_env);
            if (nacc == 1)      { launch_shape<2, half, 1>(x, w, out, out_row_stride, stream); }
            else if (nacc == 4) { launch_shape<2, half, 4>(x, w, out, out_row_stride, stream); }
            else                { launch_shape<2, half>(x, w, out, out_row_stride, stream); }
        }
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
