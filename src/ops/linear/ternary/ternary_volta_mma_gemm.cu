#include "core/device.h"
#include "ops/common/volta_mma_splits.h"
#include "ops/linear/ternary/ternary_launch.h"
#include "ops/linear/ternary/ternary_volta_mma_gemm.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

// Shape sweep. FLOP/byte is 8*kTSub/(kTSub+1) and is independent of kWarps, so kTSub is the only
// ratio lever and kWarps is purely an occupancy lever -- but the shared footprint grows with both,
// so the two have to be traded deliberately. `min_blocks` is chosen per variant as the largest CTA
// count that still fits 96 KiB of shared, capped where register pressure starts to bind (the kernel
// needs roughly 60 registers per thread; 65536/(threads*min_blocks) is the budget it gets).
//
//   variant  warps  tsub  shared/CTA  CTAs/SM  threads/SM  occupancy  FLOP/byte
//   c4t1       4     1     10240        8        1024       50.0%        4.0
//   c4t2       4     2     15360        6         768       37.5%        5.33
//   c4t4       4     4     25600        3         384       18.75%       6.4
//   c8t2       8     2     20480        4        1024       50.0%        5.33
//   c8t4       8     4     30720        3         768       37.5%       6.4
//   c4t8       4     8     46080        2         256       12.5%       7.11
//   c8t8       8     8     51200        1         256       12.5%       7.11
//
// c8t4 is the shape that gets c4t4's traffic ratio at twice its occupancy, which is why it is the
// default; c4t1 is the original port and stays as the baseline arm.
//
// c8t8 is deliberately NOT instantiated. 51200 B is over the 48 KiB static shared limit, so it
// would need cudaFuncSetAttribute plus dynamic shared memory (the V100 does allow 96 KiB opt-in).
// It is dominated anyway -- c4t8 reaches the same FLOP/byte with a smaller footprint and one more
// resident CTA -- so the opt-in path buys nothing and is not worth the launch-time tax.
struct TernaryMmaVariantDesc {
    const char* name;
    int warps;
    int tsub;
    int min_blocks;
};

// Preference order: best throughput at long T first. Each entry's token tile is
// kTernaryMmaSubTile * tsub, and the shape only wins if the prompt fills at least kMinTiles of
// them -- otherwise the last CTA is mostly padding and the wider tile is a pure loss. Measured at
// T=164: c4t1 346.5 t/s against c8t4 167.8, because a 128-token tile leaves 72% of its second CTA
// empty. At T=612 the same two are 437.0 and 670.5. Four tiles keeps the waste under 25%.
inline constexpr int kTernaryMmaMinTiles = 4;

inline constexpr TernaryMmaVariantDesc kTernaryMmaOrder[] = {
    {"c8t4", 8, 4, 3}, {"c8t2", 8, 2, 4}, {"c4t4", 4, 4, 3},
    {"c4t2", 4, 2, 6}, {"c4t1", 4, 1, 8}, {"c4t8", 4, 8, 2},
};

const TernaryMmaVariantDesc& forced_variant() {
    static const TernaryMmaVariantDesc* chosen = []() -> const TernaryMmaVariantDesc* {
        const char* value = std::getenv("NINFER_TERNARY_MMA");
        if (value == nullptr) { return nullptr; }
        const std::string text(value);
        if (text == "auto") { return nullptr; }
        for (const TernaryMmaVariantDesc& v : kTernaryMmaOrder) {
            if (text == v.name) { return &v; }
        }
        throw std::invalid_argument(
            "NINFER_TERNARY_MMA=" + text +
            " is not a known shape (auto, c4t1, c4t2, c4t4, c4t8, c8t2, c8t4)");
    }();
    return chosen == nullptr ? kTernaryMmaOrder[4] : *chosen;
}

bool variant_forced() noexcept {
    const char* value = std::getenv("NINFER_TERNARY_MMA");
    return value != nullptr && std::string(value) != "auto";
}

// The shape this token count actually launches. Every caller -- admission and launch -- goes
// through here so the split-count decision and the kernel cannot disagree about the tile size.
const TernaryMmaVariantDesc& variant_for(std::int32_t t) {
    const TernaryMmaVariantDesc& forced = forced_variant();
    if (variant_forced()) { return forced; }
    for (const TernaryMmaVariantDesc& v : kTernaryMmaOrder) {
        if (t >= kTernaryMmaMinTiles * kTernaryMmaSubTile * v.tsub) { return v; }
    }
    return kTernaryMmaOrder[4];
}

// Split count. Same policy as volta_mma_split_count(), except the token tile is
// kTernaryMmaSubTile * tsub rather than the host-side kVoltaMmaTTile, so a larger tsub makes each
// CTA cover more tokens and therefore produces fewer CTAs.
int ternary_volta_mma_split_count(std::int32_t n, std::int32_t k, std::int32_t t, int warps,
                                  int tsub) noexcept {
    const int rows_per_cta = warps * 8;
    const int row_ctas = (n + rows_per_cta - 1) / rows_per_cta;
    const int t_tile   = kTernaryMmaSubTile * tsub;
    const int t_tiles  = (t + t_tile - 1) / t_tile;
    const int blocks   = row_ctas * t_tiles;
    int splits = blocks >= kVoltaMmaCtaBudget ? 1 : kVoltaMmaCtaBudget / std::max(blocks, 1);
    if (k >= kVoltaMmaLongK) { splits = std::min(splits, 2); }
    return std::max(splits, 1);
}

// Admission for the tensor-core route.
//
// `splits == 1` is a hard requirement, not a preference: the split-K path accumulates into an fp32
// [n, t] workspace with atomicAdd and then runs a narrowing pass, and TernaryLaunch -- the function
// pointer the whole ternary dispatch chain goes through -- carries no WorkspaceArena. Rather than
// re-plumb that signature for a case this model never hits, the route is refused when the split
// count is above one and the caller falls through to the SIMT routes. For prefill that is
// automatic: T=3412 at c8t4 gives 1088 row CTAs x 27 token tiles against a 384-CTA budget.
bool ternary_volta_mma_admits(const Tensor& x, const Weight& w) noexcept {
    constexpr int kGroupK = PQ2RowSplitStorage::kGroupK;
    if (w.qtype != QType::PQ2_0_G128) { return false; }
    if (w.qhigh != nullptr || w.padded_shape[1] != w.k) { return false; }
    if (w.k <= 0 || w.n <= 0) { return false; }
    if (x.ne[0] != w.k) { return false; }
    if (w.k % kGroupK != 0 || w.k % 8 != 0) { return false; }
    // Below one whole A tile the route has nothing to offer; the GEMV/tile kernels own that band.
    if (x.ne[1] < kTernaryMmaSubTile) { return false; }
    // Each CTA needs at least one whole kStep of K.
    if (w.k < kTernaryMmaKStep) { return false; }
    return ternary_volta_mma_split_count(w.n, w.k, x.ne[1], variant_for(x.ne[1]).warps,
                                         variant_for(x.ne[1]).tsub) == 1;
}

namespace {

template <class Shape, int kMinBlocks, int kProbe = kProbeOff, int kCommit = kCommitTrailing>
void launch_shape(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_ld,
                  cudaStream_t stream) {
    static_assert(Shape::kSharedBytes * kMinBlocks <= kVoltaSharedPerSm,
                  "shape does not fit the shared budget at this occupancy");
    static_assert(Shape::kThreads * kMinBlocks <= kVoltaThreadsPerSm,
                  "shape does not fit the thread budget at this occupancy");
    static_assert(Shape::kSharedBytes <= 48 * 1024,
                  "static shared is capped at 48 KiB; a larger shape needs the opt-in dynamic path");
    const std::int32_t n = w.n;
    const std::int32_t k = w.k;
    const std::int32_t t = x.ne[1];
    const std::int32_t padded_groups = w.padded_shape[1] / PQ2RowSplitStorage::kGroupK;
    const dim3 grid(static_cast<unsigned>((n + Shape::kRowsPerCta - 1) / Shape::kRowsPerCta), 1u,
                    static_cast<unsigned>((t + Shape::kTTile - 1) / Shape::kTTile));
    // splits == 1 is guaranteed by the admission predicate, so this is always the direct variant:
    // no fp32 workspace, no memset, no atomics, no narrowing pass.
    ternary_volta_mma_gemm_kernel<Shape, kMinBlocks, true, kProbe, kCommit>
        <<<grid, Shape::kThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
            static_cast<const __nv_bfloat16*>(x.data), nullptr,
            static_cast<__nv_bfloat16*>(out.data), out_ld, n, k, t, padded_groups, 1);
    CUDA_CHECK(cudaGetLastError());
}

// Shape selection and commit placement are orthogonal knobs, so they compose here rather than
// overriding each other. Every (shape, commit) pair is instantiated; the sweep can then vary one
// while the other sits at its default.
template <int kProbe, int kCommit>
void launch_selected(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_ld,
                     cudaStream_t stream) {
    const std::string name(variant_for(x.ne[1]).name);
    if (name == "c4t1") {
        launch_shape<TernaryVoltaMmaShape<4, 1>, 8, kProbe, kCommit>(x, w, out, out_ld, stream);
    } else if (name == "c4t2") {
        launch_shape<TernaryVoltaMmaShape<4, 2>, 6, kProbe, kCommit>(x, w, out, out_ld, stream);
    } else if (name == "c4t4") {
        launch_shape<TernaryVoltaMmaShape<4, 4>, 3, kProbe, kCommit>(x, w, out, out_ld, stream);
    } else if (name == "c8t2") {
        launch_shape<TernaryVoltaMmaShape<8, 2>, 4, kProbe, kCommit>(x, w, out, out_ld, stream);
    } else if (name == "c4t8") {
        launch_shape<TernaryVoltaMmaShape<4, 8>, 2, kProbe, kCommit>(x, w, out, out_ld, stream);
    } else {
        launch_shape<TernaryVoltaMmaShape<8, 4>, 3, kProbe, kCommit>(x, w, out, out_ld, stream);
    }
}

} // namespace

void launch_ternary_volta_mma(const Tensor& x, const Weight& w, Tensor& out,
                              std::int32_t out_row_stride, cudaStream_t stream) {
    if (out_row_stride < w.n) {
        throw std::invalid_argument("ternary volta mma: output row stride is smaller than the tile");
    }
    // NINFER_TERNARY_MMA_PROBE=nodecode / noaload selects the timing probe described at the kernel
    // template. Its output is deliberately wrong; it is a measurement, not a route.
    static const int probe = [] {
        const char* value = std::getenv("NINFER_TERNARY_MMA_PROBE");
        if (value == nullptr) { return kProbeOff; }
        const std::string text(value);
        if (text == "nodecode") { return kProbeNoDecode; }
        if (text == "noaload") { return kProbeNoALoad; }
        return kProbeOff;
    }();
    if (probe != kProbeOff) {
        if (probe == kProbeNoDecode) {
            launch_shape<TernaryVoltaMmaShape<8, 4>, 3, kProbeNoDecode>(x, w, out, out_row_stride,
                                                                       stream);
        } else {
            launch_shape<TernaryVoltaMmaShape<8, 4>, 3, kProbeNoALoad>(x, w, out, out_row_stride,
                                                                      stream);
        }
        return;
    }
    // NINFER_TERNARY_MMA_COMMIT=early|split moves the weight decode off the critical path; unset
    // keeps early, the measured winner (trailing 709.6, early 734.2, split 725.1 -- all
    // token-identical). It is orthogonal to NINFER_TERNARY_MMA, so the two compose.
    static const std::string commit_mode = [] {
        const char* value = std::getenv("NINFER_TERNARY_MMA_COMMIT");
        if (value == nullptr) { return std::string("early"); }
        const std::string text(value);
        if (text == "split" || text == "trailing" || text == "early") { return text; }
        return std::string("early");
    }();
    if (commit_mode == "split") {
        launch_selected<kProbeOff, kCommitSplit>(x, w, out, out_row_stride, stream);
    } else if (commit_mode == "trailing") {
        launch_selected<kProbeOff, kCommitTrailing>(x, w, out, out_row_stride, stream);
    } else {
        launch_selected<kProbeOff, kCommitEarly>(x, w, out, out_row_stride, stream);
    }
}

#else

bool ternary_volta_mma_admits(const Tensor&, const Weight&) noexcept { return false; }

void launch_ternary_volta_mma(const Tensor&, const Weight&, Tensor&, std::int32_t, cudaStream_t) {
    throw std::invalid_argument("ternary volta mma: this build has no Volta tensor-core route");
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
