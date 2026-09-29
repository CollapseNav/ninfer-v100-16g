// Ported to the Volta (sm_70) tree from the Ambolio/ninfer-4090-windows ternary lineage.
// The original banner read "Ada sm_89"; this copy is built and verified for sm_70 only.
//
// Route selection here is the SIMT family ONLY. The author's tree picks between five kernels on
// the group geometry and token count; four of them are bf16/int8 TENSOR-CORE schedules that Volta
// cannot run at all (no bf16 MMA before sm_80; the int8 rung wants m16n8k32.s8 which is Ampere+).
// The ones it can run are the warp-shuffle GEMV family in ternary_rowsplit_gemv.cuh and the
// correctness-first reference kernel in ternary_rowsplit_gemm.cuh; both are pure CUDA-core code.
//
// So the sm_70 route table is:
//
//   T == 1        warp-per-row GEMV                 (one 32-byte code span + 2-byte scale per group)
//   T == 2..4     token-tile GEMV, weights read once for all tokens (the MTP verify shape)
//   T >= 5        row-blocked GEMV                  (row block amortises activations over rows)
//   any T         reference kernel, NINFER_TERNARY_PREFILL=ref forces it (the oracle arm)
//
// Every tensor-core arm is compiled out under NINFER_VOLTA_BUILD rather than left to fail at
// launch: including the headers at all pulls in mma PTX and bf16 arithmetic helpers that do not
// exist for sm_70, so the guard has to be at the include, not at the call site.
//
// Consequence to keep in mind while measuring: prefill above ~64 tokens is issue-bound on the
// SIMT path (the author measured MMA 8.3-9.8x faster than the blocked GEMV at T=1024 on Ada).
// Decode and the MTP verify pass are NOT affected -- they were already SIMT on the author's card.
#include "ops/linear/ternary/ternary_rowsplit_gemm.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/ternary/ternary_launch.h"
#include "ops/linear/ternary/ternary_rowsplit_gemm_simt.cuh"
#include "ops/linear/ternary/ternary_rowsplit_gemv.cuh"
#ifndef NINFER_VOLTA_BUILD
#include "ops/linear/ternary/ternary_rowsplit_mma.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_small_t.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_wide_t.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_s8.cuh"
#endif
#include "ops/linear/ternary/ternary_s8_scratch.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

// Which kernel serves prefill (T >= 5).
//
//   unset / mma    the PQ2 fp16 tensor-core GEMM (ternary_volta_mma_gemm.cuh) -- the measured
//                  default: 453 t/s against 110 for the blocked GEMV and 115 for the SIMT GEMM on
//                  a 3412-token prompt, with byte-identical greedy token ids
//   block          the row-blocked SIMT GEMV, the previous default, kept as the A/B arm
//   simt           the PQ2 word-parallel SIMT GEMM (ternary_rowsplit_gemm_simt.cuh)
//   ref            the correctness-first reference kernel, the oracle for the fast path
//
// A two-run A/B against `ref` is what qualifies a route: it is the only comparison that can catch a
// token-tile or activation-layout error, because at T == 1 the token-major and row-major activation
// layouts coincide exactly. Read once, because the choice decides which kernel enters a captured
// CUDA graph.
//
// `wide` is still refused rather than remapped: it names the author's bf16 weight-resident kernel,
// which does not exist here, and silently aliasing it to a SIMT route would make an A/B run against
// the author's numbers look like a regression in that kernel family when it is really a missing
// route. `mma` is no longer in that category -- there is now a real fp16 tensor-core kernel behind
// it -- but it still refuses at launch time for shapes it cannot serve (see
// ternary_volta_mma_admits), and those fall through to the SIMT routes below rather than throwing.
// That is why making it the default is safe: below one 32-token A tile, and on shapes whose split-K
// count exceeds one, the arms below still carry the work.
enum class PrefillRoute { Block, Simt, Mma, Reference };

PrefillRoute prefill_route() {
    static const PrefillRoute route = [] {
        const char* value = std::getenv("NINFER_TERNARY_PREFILL");
        if (value == nullptr) { return PrefillRoute::Mma; }
        const std::string text(value);
        if (text == "ref") { return PrefillRoute::Reference; }
        if (text == "simt") { return PrefillRoute::Simt; }
        if (text == "block") { return PrefillRoute::Block; }
        if (text == "mma") { return PrefillRoute::Mma; }
        if (text == "wide") {
            throw std::invalid_argument(
                "NINFER_TERNARY_PREFILL=wide selects the author's bf16 weight-resident kernel, "
                "which does not exist on sm_70; use mma (default), block, simt or ref");
        }
        return PrefillRoute::Mma;
    }();
    return route;
}

// T=1 dispatch with an env-selectable warps-per-block. NINFER_TERNARY_GEMV_WARPS picks one of
// 4 / 8 / 16 / 32; anything else keeps the original 8. Read once, because the choice decides which
// kernel enters a captured CUDA graph.
template <int kWarps, bool kSkipBias, int kUnroll = 1, int kProbe = kGemvProbeOff, int kRows = 1,
          int kMinBlocks = 1>
void launch_gemv_w(const Tensor& x, const Weight& w, Tensor& out, std::int32_t groups_per_row,
                   cudaStream_t stream) {
    const unsigned grid = static_cast<unsigned>(div_up(w.n, kWarps * kRows));
    ternary_pq2_gemv_w_kernel<kWarps, kSkipBias, kUnroll, kProbe, kRows, kMinBlocks>
        <<<grid, kWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
            groups_per_row);
    CUDA_CHECK(cudaGetLastError());
}

// NINFER_TERNARY_GEMV_ROWS=1|2|4 holds that many output rows per warp. The activation window is the
// same for every row, so this is the only change that attacks both the per-weight instruction count
// and the per-warp memory-level parallelism at once.
int gemv_rows() {
    static const int rows = [] {
        const char* value = std::getenv("NINFER_TERNARY_GEMV_ROWS");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return (parsed == 1 || parsed == 2 || parsed == 4) ? parsed : 1;
    }();
    return rows;
}

// NINFER_TERNARY_GEMV_PROBE=noact|nocode|noscale deletes one load class from the T=1 kernel at a
// time. Its output is deliberately wrong; it exists because the standalone read-pattern benchmark
// (gemv_probe.cu) already cleared the code walk -- 869.6 GB/s against an 882.6 GB/s flat ceiling --
// so whatever costs the real kernel its 3x has to be located in the real kernel.
int gemv_probe() {
    static const int probe = [] {
        const char* value = std::getenv("NINFER_TERNARY_GEMV_PROBE");
        if (value == nullptr) { return kGemvProbeOff; }
        const std::string text(value);
        if (text == "noact") { return kGemvNoActivation; }
        if (text == "nocode") { return kGemvNoCode; }
        if (text == "noscale") { return kGemvNoScale; }
        if (text == "codealu") { return kGemvCodeFromAlu; }
        return kGemvProbeOff;
    }();
    return probe;
}

// NINFER_TERNARY_GEMV_NOBIAS=1 selects the instruction-count probe described at the kernel. Its
// output is deliberately wrong; it is a timing measurement, not a route.
// NINFER_TERNARY_GEMV_UNROLL picks the group-loop unroll factor: 1 (default), 2, 4 or 8.
int unroll_factor() {
    static const int factor = [] {
        const char* value = std::getenv("NINFER_TERNARY_GEMV_UNROLL");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8 || parsed == 12 ||
                parsed == 16 || parsed == 20)
                   ? parsed
                   : 8;
    }();
    return factor;
}

bool gemv_nobias_probe() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_GEMV_NOBIAS");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
}

// NINFER_TERNARY_GEMV_MINBLOCKS=1|2|3|4|5|6|8 picks __launch_bounds__'s second argument, which caps
// registers at 65536/(256*kMinBlocks). At kThreads=256 that is 256/128/85/64/51/42/32 registers.
//
// MEASURED. The cap only bites from 4 up:
//   minBlocks  1    2    3    4    5    6    8
//   registers 72   72   72   62   48   40   32
//   warps/SM  28   28   28   32   42   50   64
//   decode  41.6 41.6 41.5 43.6 42.5 38.5 33.5   t/s, 2 reps each, identical
// 4 wins, so it is the default: +4.8% for a build-time knob with no semantic change (greedy token
// ids are unchanged at every value). Below 48 registers it falls off, which is the unroll-8 arm
// running out of room for its eight in-flight loads.
int gemv_min_blocks() {
    static const int value = [] {
        const char* env   = std::getenv("NINFER_TERNARY_GEMV_MINBLOCKS");
        const int parsed  = env == nullptr ? 0 : std::atoi(env);
        switch (parsed) {
        case 1: case 2: case 3: case 5: case 6: case 8: return parsed;
        case 4: return 4;
        default: return 4;   // measured default
        }
    }();
    return value;
}

void launch_gemv_t1(const Tensor& x, const Weight& w, Tensor& out, std::int32_t groups_per_row,
                    cudaStream_t stream) {
    static const int warps = [] {
        const char* value = std::getenv("NINFER_TERNARY_GEMV_WARPS");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return (parsed == 4 || parsed == 8 || parsed == 16 || parsed == 32) ? parsed : 8;
    }();
    if (gemv_nobias_probe()) {
        launch_gemv_w<8, true>(x, w, out, groups_per_row, stream);
        return;
    }
    switch (gemv_probe()) {
    case kGemvNoActivation:
        launch_gemv_w<8, false, 8, kGemvNoActivation>(x, w, out, groups_per_row, stream);
        return;
    case kGemvNoCode:
        launch_gemv_w<8, false, 8, kGemvNoCode>(x, w, out, groups_per_row, stream);
        return;
    case kGemvNoScale:
        launch_gemv_w<8, false, 8, kGemvNoScale>(x, w, out, groups_per_row, stream);
        return;
    case kGemvCodeFromAlu:
        launch_gemv_w<8, false, 8, kGemvCodeFromAlu>(x, w, out, groups_per_row, stream);
        return;
    default: break;
    }
    switch (gemv_rows()) {
    case 2: launch_gemv_w<8, false, 8, kGemvProbeOff, 2>(x, w, out, groups_per_row, stream); return;
    case 4: launch_gemv_w<8, false, 8, kGemvProbeOff, 4>(x, w, out, groups_per_row, stream); return;
    default: break;
    }
    // Register-cap arm. 4 is the measured default; the others exist only to re-run the sweep, and
    // each is instantiated at unroll 8 alone.
    const int min_blocks = gemv_min_blocks();
    if (min_blocks != 4) {
        switch (min_blocks) {
        case 1: launch_gemv_w<8, false, 8, kGemvProbeOff, 1, 1>(x, w, out, groups_per_row, stream); return;
        case 2: launch_gemv_w<8, false, 8, kGemvProbeOff, 1, 2>(x, w, out, groups_per_row, stream); return;
        case 3: launch_gemv_w<8, false, 8, kGemvProbeOff, 1, 3>(x, w, out, groups_per_row, stream); return;
        case 5: launch_gemv_w<8, false, 8, kGemvProbeOff, 1, 5>(x, w, out, groups_per_row, stream); return;
        case 6: launch_gemv_w<8, false, 8, kGemvProbeOff, 1, 6>(x, w, out, groups_per_row, stream); return;
        default: launch_gemv_w<8, false, 8, kGemvProbeOff, 1, 8>(x, w, out, groups_per_row, stream); return;
        }
    }
    // Re-sweep the unroll INSIDE the winning cap. Unroll 12/16/20 lost on their own because ptxas
    // spent 95/116/134 registers on them; at a 64-register ceiling the extra in-flight loads may pay.
    switch (unroll_factor()) {
    case 2:  launch_gemv_w<8, false, 2,  kGemvProbeOff, 1, 4>(x, w, out, groups_per_row, stream); return;
    case 4:  launch_gemv_w<8, false, 4,  kGemvProbeOff, 1, 4>(x, w, out, groups_per_row, stream); return;
    case 12: launch_gemv_w<8, false, 12, kGemvProbeOff, 1, 4>(x, w, out, groups_per_row, stream); return;
    case 16: launch_gemv_w<8, false, 16, kGemvProbeOff, 1, 4>(x, w, out, groups_per_row, stream); return;
    case 20: launch_gemv_w<8, false, 20, kGemvProbeOff, 1, 4>(x, w, out, groups_per_row, stream); return;
    default: break;
    }
    // The warps-per-block arm deliberately gets no cap: at 512 and 1024 threads a min-blocks of 4
    // would demand 32 and 16 registers and spill the kernel to death. It is flat-to-inverse anyway.
    switch (warps) {
    case 4:  launch_gemv_w<4,  false>(x, w, out, groups_per_row, stream); return;
    case 16: launch_gemv_w<16, false>(x, w, out, groups_per_row, stream); return;
    case 32: launch_gemv_w<32, false>(x, w, out, groups_per_row, stream); return;
    default: launch_gemv_w<8, false, 8, kGemvProbeOff, 1, 4>(x, w, out, groups_per_row, stream); return;
    }
}

// Decode (T == 1) takes the warp-per-row GEMV for PQ2_0. K is a whole number of 128-groups for
// every width in this model, so that kernel needs no column guard.
void launch_pq2_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if ((w.k % 128) != 0 || x.ne[1] != 1) {
        throw std::invalid_argument("ternary gemv: expected one token and a whole-group K");
    }
    const std::int32_t groups_per_row = w.k / 128;
    launch_gemv_t1(x, w, out, groups_per_row, stream);
}

// NINFER_TERNARY_FP16_ACT=2 selects the bit-exact fp16 variant (container only, fp32 dot chain);
// =1 selects the half2 dot, which rounds the group sum once to fp16.
bool fp16_mode2() {
    static const bool value = [] {
        const char* env = std::getenv("NINFER_TERNARY_FP16_ACT");
        return env != nullptr && std::string(env) == "2";
    }();
    return value;
}

// Small-token-tile GEMV: weights are read once for up to 4 tokens, which is what makes the
// speculative verify pass (T = draft + 1) cheap. Falls back to the reference tiled kernel beyond
// that, and for PTQ1_0 / padded-K weights.
void launch_pq2_gemv_tile(const Tensor& x, const Weight& w, Tensor& out,
                          std::int32_t out_row_stride, std::int32_t tokens,
                          cudaStream_t stream) {
    if ((w.k % 128) != 0) {
        throw std::invalid_argument("ternary gemv: K must be a whole number of 128-groups");
    }
    const std::int32_t groups_per_row = w.k / 128;
    const unsigned grid               = static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock));
    const dim3 block(kGemvWarpsPerBlock * 32, 1u, 1u);
    if (tokens <= 1) {
        launch_gemv_t1(x, w, out, groups_per_row, stream);
    } else {
        // kT follows the real token count. Instantiating <4> for every verify measured 1.283 ms at
        // two live tokens against 1.024 ms for a real kT=2, so the guard waste is a fifth of the pass.
        //
        // The unroll is a separate knob because one extra verified token costs 0.60 of a whole T=1
        // step, and this loop had no pragma at all: ptxas chose for itself while the per-token work
        // (a load, two converts and five FMAs per token per group) was layered on top of it. The T=1
        // GEMV's own curve is 1 -> 29.9, 2 -> 34.1, 4 -> 40.4, 8 -> 41.6, 12 -> 33.9 t/s, so the
        // depth is worth measuring rather than inheriting. 4 is the unmeasured starting point.
        static const int unroll = [] {
            const char* value = std::getenv("NINFER_TERNARY_GEMV_TILE_UNROLL");
            const int parsed  = value == nullptr ? 0 : std::atoi(value);
            return (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8 || parsed == 16) ? parsed
                                                                                               : 0;
        }();
        // Measured on prose with MTP K=1..3: the two-token tile wants depth 8 (51.8 t/s against
        // 50.5 at 4), the four-token tile wants 4 (42.0 against 38.8 at 8). One default per kT is
        // therefore the honest encoding of the sweep, and the env still overrides both.
        const int depth = unroll != 0 ? unroll : (tokens == 2 ? 8 : 4);
        const auto launch_tile = [&](auto token_tag, auto unroll_tag) {
            using TokenTag  = decltype(token_tag);
            using UnrollTag = decltype(unroll_tag);
            if (x.dtype == DType::FP16) {
                if (fp16_mode2()) {
                    ternary_pq2_gemv_tile_kernel<TokenTag::value, UnrollTag::value, false, 2>
                        <<<grid, block, 0, stream>>>(
                            static_cast<const __nv_bfloat16*>(x.data),
                            static_cast<const std::uint8_t*>(w.qdata),
                            static_cast<const std::uint8_t*>(w.scales),
                            static_cast<__nv_bfloat16*>(out.data), w.n, groups_per_row, tokens,
                            out_row_stride);
                } else {
                    ternary_pq2_gemv_tile_kernel<TokenTag::value, UnrollTag::value, false, 1>
                        <<<grid, block, 0, stream>>>(
                            static_cast<const __nv_bfloat16*>(x.data),
                            static_cast<const std::uint8_t*>(w.qdata),
                            static_cast<const std::uint8_t*>(w.scales),
                            static_cast<__nv_bfloat16*>(out.data), w.n, groups_per_row, tokens,
                            out_row_stride);
                }
            } else {
                ternary_pq2_gemv_tile_kernel<TokenTag::value, UnrollTag::value, false, 0>
                    <<<grid, block, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(x.data),
                        static_cast<const std::uint8_t*>(w.qdata),
                        static_cast<const std::uint8_t*>(w.scales),
                        static_cast<__nv_bfloat16*>(out.data), w.n, groups_per_row, tokens,
                        out_row_stride);
            }
        };
        using std::integral_constant;
        // kMaxKt is 16 -- the tile kernel serves the whole verify band, including the 15-token
        // context-lookup window that makes the verify pass T = 16 -- unless
        // NINFER_TERNARY_TILE_WIDE=0 hands T >= 5 back to the row-blocked kernel.
        static const int kMaxKt = [] {
            const char* value = std::getenv("NINFER_TERNARY_TILE_WIDE");
            return (value != nullptr && std::string(value) == "0") ? 4 : 16;
        }();
        const std::int32_t kt = tokens < kMaxKt ? tokens : kMaxKt;
        // PROBE: NINFER_TERNARY_TILE_SHARE_ACT=1 runs the tile kernel with every token reusing the
        // first token's activation values, so the output is wrong by construction. It prices the
        // per-token activation side of the loop -- the loads and the bf16->fp32 converts -- against
        // the rest of the body. Never a default.
        static const bool share_act = [] {
            const char* env = std::getenv("NINFER_TERNARY_TILE_SHARE_ACT");
            return env != nullptr && std::string(env) != "0";
        }();
        if (share_act) {
            const auto launch_probe = [&](auto token_tag) {
                constexpr int kKtProbe = decltype(token_tag)::value;
                constexpr int kDepthProbe = kKtProbe == 2 ? 8 : 4;
                ternary_pq2_gemv_tile_kernel<kKtProbe, kDepthProbe, true>
                    <<<grid, block, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(x.data),
                        static_cast<const std::uint8_t*>(w.qdata),
                        static_cast<const std::uint8_t*>(w.scales),
                        static_cast<__nv_bfloat16*>(out.data), w.n, groups_per_row, tokens,
                        out_row_stride);
            };
            switch (kt) {
            case 2: launch_probe(integral_constant<int, 2>{}); break;
            case 4: launch_probe(integral_constant<int, 4>{}); break;
            case 8: launch_probe(integral_constant<int, 8>{}); break;
            case 16: launch_probe(integral_constant<int, 16>{}); break;
            default: launch_probe(integral_constant<int, 4>{}); break;
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        switch (kt) {
        case 16:
            if (depth == 1) { launch_tile(integral_constant<int, 16>{}, integral_constant<int, 1>{}); }
            else if (depth == 2) { launch_tile(integral_constant<int, 16>{}, integral_constant<int, 2>{}); }
            else if (depth == 8) { launch_tile(integral_constant<int, 16>{}, integral_constant<int, 8>{}); }
            else if (depth == 16) { launch_tile(integral_constant<int, 16>{}, integral_constant<int, 16>{}); }
            else { launch_tile(integral_constant<int, 16>{}, integral_constant<int, 4>{}); }
            break;
        case 5:
            launch_tile(integral_constant<int, 5>{}, integral_constant<int, 4>{});
            break;
        case 6:
            launch_tile(integral_constant<int, 6>{}, integral_constant<int, 4>{});
            break;
        case 7:
            launch_tile(integral_constant<int, 7>{}, integral_constant<int, 4>{});
            break;
        case 8:
            if (depth == 1) { launch_tile(integral_constant<int, 8>{}, integral_constant<int, 1>{}); }
            else if (depth == 2) { launch_tile(integral_constant<int, 8>{}, integral_constant<int, 2>{}); }
            else if (depth == 8) { launch_tile(integral_constant<int, 8>{}, integral_constant<int, 8>{}); }
            else if (depth == 16) { launch_tile(integral_constant<int, 8>{}, integral_constant<int, 16>{}); }
            else { launch_tile(integral_constant<int, 8>{}, integral_constant<int, 4>{}); }
            break;
        case 2:
            if (depth == 1) { launch_tile(integral_constant<int, 2>{}, integral_constant<int, 1>{}); }
            else if (depth == 2) { launch_tile(integral_constant<int, 2>{}, integral_constant<int, 2>{}); }
            else if (depth == 4) { launch_tile(integral_constant<int, 2>{}, integral_constant<int, 4>{}); }
            else if (depth == 16) { launch_tile(integral_constant<int, 2>{}, integral_constant<int, 16>{}); }
            else { launch_tile(integral_constant<int, 2>{}, integral_constant<int, 8>{}); }
            break;
        case 3:
            if (depth == 1) { launch_tile(integral_constant<int, 3>{}, integral_constant<int, 1>{}); }
            else if (depth == 2) { launch_tile(integral_constant<int, 3>{}, integral_constant<int, 2>{}); }
            else if (depth == 8) { launch_tile(integral_constant<int, 3>{}, integral_constant<int, 8>{}); }
            else if (depth == 16) { launch_tile(integral_constant<int, 3>{}, integral_constant<int, 16>{}); }
            else { launch_tile(integral_constant<int, 3>{}, integral_constant<int, 4>{}); }
            break;
        default:
            if (depth == 1) { launch_tile(integral_constant<int, 4>{}, integral_constant<int, 1>{}); }
            else if (depth == 2) { launch_tile(integral_constant<int, 4>{}, integral_constant<int, 2>{}); }
            else if (depth == 8) { launch_tile(integral_constant<int, 4>{}, integral_constant<int, 8>{}); }
            else if (depth == 16) { launch_tile(integral_constant<int, 4>{}, integral_constant<int, 16>{}); }
            else { launch_tile(integral_constant<int, 4>{}, integral_constant<int, 4>{}); }
            break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Storage, class Atom, int kTileT>
void launch_gemm(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                 cudaStream_t stream) {
    const std::int32_t rows = w.n;
    const std::int32_t k    = w.k;
    const std::int32_t t    = x.ne[1];
    if (k % Storage::kGroupK != 0) {
        throw std::invalid_argument("ternary linear: K must be a multiple of the group size");
    }
    if (out_row_stride < rows) {
        throw std::invalid_argument("ternary linear: output row stride is smaller than the tile");
    }
    const std::int32_t groups_per_row = k / Storage::kGroupK;

    const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(div_up(t, kTileT)), 1u);
    constexpr dim3 block(Storage::kGroupK, 1u, 1u);

    ternary_rowsplit_gemm_kernel<Storage, Atom, kTileT><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), rows, k, t, groups_per_row, out_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

template <int kTileT>
void launch_by_qtype(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                     cudaStream_t stream) {
    switch (w.qtype) {
    case QType::PTQ1_0_G128:
        launch_gemm<PTQ1RowSplitStorage, PTQ1SimtDecodeAtom, kTileT>(x, w, out, out_row_stride,
                                                                    stream);
        return;
    case QType::PQ2_0_G128:
        launch_gemm<PQ2RowSplitStorage, PQ2SimtDecodeAtom, kTileT>(x, w, out, out_row_stride,
                                                                  stream);
        return;
    default:
        break;
    }
    throw std::invalid_argument("ternary linear: unsupported weight qtype");
}

} // namespace

// A PQ2_0 weight can take the fast family whenever the layout did not pad K past the real width --
// true for every width in this model (5120/6144/10240/17408 are all whole 128-groups), and checked
// here rather than assumed, because these kernels read whole groups without a column guard.
//
// Every one of them walks the activation as x[token * w.k + column], so the x row pitch has to BE
// w.k. That holds for both callers today (the raw path's hidden and the folded activation both
// report w.k as ne[0]), but it is assumed rather than enforced anywhere else, and a mismatch would
// read the wrong rows instead of failing. Checked here so all three routes agree on the gate.
bool gemv_admits(const Tensor& x, const Weight& w, std::int32_t max_tokens) {
    return w.qtype == QType::PQ2_0_G128 && w.qhigh == nullptr && w.padded_shape[1] == w.k &&
           (w.k % 128) == 0 && x.ne[1] >= 1 && x.ne[1] <= max_tokens && x.ne[0] == w.k;
}

// Routing switch for the speculative VERIFY band (T = 2..4), which the small-tile GEMV serves by
// default. That kernel reads each weight byte once for every token in the tile, but it re-reads the
// ACTIVATION for every output row: per group per lane it loads 2*kT activation elements against one
// weight byte. The row-blocked kernel below amortises the activation over kR output rows
// (loads per FMA = (2*kT + kR) / (4*kR*kT): 0.56 at kR=1,kT=4 against 0.19 at kR=4,kT=4), and it is
// already the shape used for T = 5..8.
//
// Routing T = 2..4 to it LOST on the Ada card the author tuned this on (MTP decode 43.0 -> 32.8 t/s)
// because there the tile kernel was issue-bound at 95% occupancy. On Volta the balance is different
// -- the tile kernel<4> compiles to 32 registers and already sits at maximum occupancy, so there is
// nothing to buy on that side and the activation traffic is all that is left. This switch exists to
// measure that rather than assume it, with the (kR, kT) pair left to NINFER_TERNARY_ROWS/_TILE.
// True (the default) when the small-tile GEMV serves the whole T = 2..8 verify band instead of
// handing T >= 5 to the row-blocked kernel. The row-blocked kernel amortises the activation over kR
// output rows, which is right in principle, but measured in situ the small-tile kernel wins the
// whole band, and by a distance that grows with the window:
//
//   K (prose)     4      5      6      7     |  K=7 on the repeated-sentence prompt
//   block        20.8   18.4   19.0   18.8   |  63.1
//   tile         38.8   29.3   28.1   25.0   |  84.0
//
// That is +87% at K=4 and +33% on the repetition workload the context-lookup path rides on.
// NINFER_TERNARY_TILE_WIDE=0 restores the row-blocked kernel for T >= 5 as the A/B arm.
bool tile_wide_enabled() {
    static const bool value = [] {
        const char* env = std::getenv("NINFER_TERNARY_TILE_WIDE");
        return env == nullptr || std::string(env) != "0";
    }();
    return value;
}

bool t2_block_enabled() {
    static const bool value = [] {
        const char* env = std::getenv("NINFER_TERNARY_T2BLOCK");
        return env != nullptr && std::string(env) != "0";
    }();
    return value;
}

void launch_ternary_gemm_t1(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch /*scratch*/) {
    if (gemv_admits(x, w, 1)) {
        launch_pq2_gemv_tile(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    if (x.dtype != DType::BF16) {
        throw std::invalid_argument(
            "ternary gemv: a non-bf16 activation reached the bf16 reference route");
    }
    launch_by_qtype<1>(x, w, out, out_row_stride, stream);
}

// One (kR, kT) instantiation of the blocked GEMV: grid.x tiles the rows in strides of
// warps*kR, grid.y tiles the tokens in kT.
template <int kR, int kT, int kUnroll = 1>
void launch_pq2_gemv_tile_block_shape(const Tensor& x, const Weight& w, Tensor& out,
                                      std::int32_t out_row_stride, std::int32_t groups_per_row,
                                      std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock * kR)),
                    static_cast<unsigned>(div_up(tokens, kT)), 1u);
    const dim3 block(kGemvWarpsPerBlock * 32, 1u, 1u);
    ternary_pq2_gemv_tile_block_kernel<kR, kT, kUnroll><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
        groups_per_row, tokens, out_row_stride);
}

// Blocked GEMV launcher, for prefill. Both block shapes are occupancy levers rather than fixed
// constants, and the measured behaviour on the target card is that the kernel is
// activation-bound, not weight-bound (R=1/kT=8 reached 125.7 t/s at an effective weight
// bandwidth of only 112 GB/s, against 422 GB/s for the T=1 GEMV). So the row block -- which
// amortises activation loads across output rows -- is the stronger knob, and the token block
// then trades weight traffic against register pressure. Both are env-selectable so one build
// can sweep; the defaults are the measured winners.
//
// NOTE for sm_70: the register budgets in these kernels were tuned on Ada. V100 has 65536
// registers per SM and the same 255-per-thread cap, but a much larger shared/L1 split and no
// cp.async, so the winning (kR, kT) here may differ from the author's 4/8. Both knobs are env
// selectable precisely so the default can be re-measured rather than assumed.
void launch_pq2_gemv_tile_block(const Tensor& x, const Weight& w, Tensor& out,
                                std::int32_t out_row_stride, cudaStream_t stream) {
    if (x.dtype != DType::BF16) {
        throw std::invalid_argument(
            "ternary gemv: the row-blocked kernel is bf16-only and received a non-bf16 activation "
            "(NINFER_TERNARY_FP16_ACT with NINFER_TERNARY_TILE_WIDE=0)");
    }
    const std::int32_t groups_per_row = w.k / 128;
    const std::int32_t tokens         = x.ne[1];
    static const int rows_block = [] {
        const char* value = std::getenv("NINFER_TERNARY_ROWS");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8) ? parsed : 4;
    }();
    static const int token_block = [] {
        const char* value = std::getenv("NINFER_TERNARY_TILE");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return (parsed == 2 || parsed == 4 || parsed == 8) ? parsed : 8;
    }();

    const auto shape = [&](auto rows_tag, auto token_tag) {
        launch_pq2_gemv_tile_block_shape<decltype(rows_tag)::value, decltype(token_tag)::value>(
            x, w, out, out_row_stride, groups_per_row, tokens, stream);
        CUDA_CHECK(cudaGetLastError());
    };
    // The shipped 4x8 shape is the one the wide verify passes use, so it gets the unroll sweep; the
    // other eleven shapes stay at the default depth to keep the (kR,kT) sweep compilable.
    const auto shape_u = [&](auto rows_tag, auto token_tag, auto unroll_tag) {
        launch_pq2_gemv_tile_block_shape<decltype(rows_tag)::value, decltype(token_tag)::value,
                                         decltype(unroll_tag)::value>(
            x, w, out, out_row_stride, groups_per_row, tokens, stream);
        CUDA_CHECK(cudaGetLastError());
    };
    static const int block_unroll = [] {
        const char* value = std::getenv("NINFER_TERNARY_BLOCK_UNROLL");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8 || parsed == 16) ? parsed : 1;
    }();
    using std::integral_constant;

    if (block_unroll != 1) {
        switch (block_unroll) {
        case 2: shape_u(integral_constant<int, 4>{}, integral_constant<int, 8>{}, integral_constant<int, 2>{}); return;
        case 4: shape_u(integral_constant<int, 4>{}, integral_constant<int, 8>{}, integral_constant<int, 4>{}); return;
        case 8: shape_u(integral_constant<int, 4>{}, integral_constant<int, 8>{}, integral_constant<int, 8>{}); return;
        case 16: shape_u(integral_constant<int, 4>{}, integral_constant<int, 8>{}, integral_constant<int, 16>{}); return;
        default: break;
        }
    }

    if (rows_block == 1 && token_block == 2) { shape(integral_constant<int, 1>{}, integral_constant<int, 2>{}); }
    else if (rows_block == 1 && token_block == 4) { shape(integral_constant<int, 1>{}, integral_constant<int, 4>{}); }
    else if (rows_block == 1) { shape(integral_constant<int, 1>{}, integral_constant<int, 8>{}); }
    else if (rows_block == 2 && token_block == 2) { shape(integral_constant<int, 2>{}, integral_constant<int, 2>{}); }
    else if (rows_block == 2 && token_block == 4) { shape(integral_constant<int, 2>{}, integral_constant<int, 4>{}); }
    else if (rows_block == 2) { shape(integral_constant<int, 2>{}, integral_constant<int, 8>{}); }
    else if (rows_block == 8 && token_block == 2) { shape(integral_constant<int, 8>{}, integral_constant<int, 2>{}); }
    else if (rows_block == 8 && token_block == 4) { shape(integral_constant<int, 8>{}, integral_constant<int, 4>{}); }
    else if (rows_block == 8) { shape(integral_constant<int, 8>{}, integral_constant<int, 8>{}); }
    else if (token_block == 2) { shape(integral_constant<int, 4>{}, integral_constant<int, 2>{}); }
    else if (token_block == 4) { shape(integral_constant<int, 4>{}, integral_constant<int, 4>{}); }
    else { shape(integral_constant<int, 4>{}, integral_constant<int, 8>{}); }
}

// --- PQ2 word-parallel SIMT GEMM ---------------------------------------------------------------
//
// kGroupsPerStage is pinned to 8 rather than to v100's Q4 choice of 16 because it has to divide the
// row's group count for a CTA to take the unpredicated `Full` path. This model's widths give 40 /
// 48 / 80 / 136 groups per row (5120 / 6144 / 10240 / 17408) and 8 is their common divisor; with 16,
// k=5120 and k=17408 would take the predicated walk and half of every third stage's staging slots
// would be zero-filled.
using TernarySimtR8C4Schedule =
    TernaryRowSplitSimtGemmSchedule<PQ2RowSplitStorage, PQ2SimtDecodeAtom, 8, 4, 8, 3, Cache::ca, 1>;
using TernarySimtR8C8Schedule =
    TernaryRowSplitSimtGemmSchedule<PQ2RowSplitStorage, PQ2SimtDecodeAtom, 8, 8, 8, 3, Cache::ca, 1>;
using TernarySimtR16C4Schedule =
    TernaryRowSplitSimtGemmSchedule<PQ2RowSplitStorage, PQ2SimtDecodeAtom, 16, 4, 8, 3, Cache::ca, 1>;
using TernarySimtR8C8P2Schedule =
    TernaryRowSplitSimtGemmSchedule<PQ2RowSplitStorage, PQ2SimtDecodeAtom, 8, 8, 8, 2, Cache::ca, 1>;
// Two ceiling probes: a wider row block (more activation reuse per staged slice) and the maximum
// legal row block. If these also land on the same number as R8C8P2 and the blocked GEMV, the limit
// is not the schedule shape.
using TernarySimtR16C8Schedule =
    TernaryRowSplitSimtGemmSchedule<PQ2RowSplitStorage, PQ2SimtDecodeAtom, 16, 8, 8, 2, Cache::ca, 1>;
using TernarySimtR32C4Schedule =
    TernaryRowSplitSimtGemmSchedule<PQ2RowSplitStorage, PQ2SimtDecodeAtom, 32, 4, 8, 2, Cache::ca, 1>;

// Admission. gemv_admits already requires PQ2_0 with no high plane, an unpadded whole-group K, and
// x.ne[0] == w.k (the activation row pitch every one of these kernels walks). On top of that the
// SIMT GEMM needs the row's group count to be a whole number of stages, which is what lets every
// CTA take `Full` and the stage walk need no predicate. Every width in this model satisfies it; a
// width that did not would fall through to the blocked GEMV below.
bool simt_admits(const Tensor& x, const Weight& w) {
    if (!gemv_admits(x, w, std::numeric_limits<std::int32_t>::max())) { return false; }
    return (w.k / PQ2RowSplitStorage::kGroupK) % TernarySimtR8C4Schedule::kGroupsPerStage == 0;
}

template <class Schedule>
void launch_ternary_simt(const Tensor& x, const Weight& w, Tensor& out,
                         std::int32_t out_row_stride, cudaStream_t stream) {
    const std::int32_t rows = w.n;
    const std::int32_t k    = w.k;
    const std::int32_t cols = x.ne[1];
    if (out_row_stride < rows) {
        throw std::invalid_argument("ternary simt: output row stride is smaller than the tile");
    }
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    // `Full` is a whole-launch property: it asserts the row block and the column tile divide
    // exactly, so nothing is predicated. The token axis is NOT sliced here (unlike the Q4 route)
    // because ColsPerTile is at most 8 and a prefill can be thousands of tokens; a short tail is
    // covered by the predicated path at the cost of one branch per column.
    const bool full =
        (rows % Schedule::kRowsPerCta) == 0 && (cols % Schedule::kColsPerTile) == 0;
    const auto* x_ptr     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes     = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales    = static_cast<const std::uint8_t*>(w.scales);
    auto* out_ptr         = static_cast<__nv_bfloat16*>(out.data);
    if (full) {
        ternary_rowsplit_gemm_simt_kernel<Schedule, true><<<grid, Schedule::kThreads, 0, stream>>>(
            x_ptr, codes, scales, out_ptr, out_row_stride, rows, k, cols, w.padded_shape[1]);
    } else {
        ternary_rowsplit_gemm_simt_kernel<Schedule, false><<<grid, Schedule::kThreads, 0, stream>>>(
            x_ptr, codes, scales, out_ptr, out_row_stride, rows, k, cols, w.padded_shape[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Occupancy levers, env-selectable so one build can sweep. All four fit the 48 KiB static shared
// budget; R16C4 exists because a wider row block amortises the activation staging further, at the
// cost of 512 threads per CTA.
void launch_ternary_simt_selected(const Tensor& x, const Weight& w, Tensor& out,
                                  std::int32_t out_row_stride, cudaStream_t stream) {
    static const int variant = [] {
        const char* value = std::getenv("NINFER_TERNARY_SIMT");
        const std::string text(value == nullptr ? "" : value);
        if (text == "r8c8") { return 1; }
        if (text == "r16c4") { return 2; }
        if (text == "r8c8p2") { return 3; }
        if (text == "r16c8") { return 4; }
        if (text == "r32c4") { return 5; }
        return 0; // r8c4
    }();
    switch (variant) {
    case 1: launch_ternary_simt<TernarySimtR8C8Schedule>(x, w, out, out_row_stride, stream); return;
    case 2: launch_ternary_simt<TernarySimtR16C4Schedule>(x, w, out, out_row_stride, stream); return;
    case 3:
        launch_ternary_simt<TernarySimtR8C8P2Schedule>(x, w, out, out_row_stride, stream);
        return;
    case 4:
        launch_ternary_simt<TernarySimtR16C8Schedule>(x, w, out, out_row_stride, stream);
        return;
    case 5:
        launch_ternary_simt<TernarySimtR32C4Schedule>(x, w, out, out_row_stride, stream);
        return;
    default:
        launch_ternary_simt<TernarySimtR8C4Schedule>(x, w, out, out_row_stride, stream);
        return;
    }
}

#ifndef NINFER_VOLTA_BUILD
// The speculative verify pass (T = 2..4) gets its own tensor-core entry point, because the prefill
// one tiles the token axis at 128 and would be 97% empty on three tokens. NINFER_TERNARY_VERIFY=tile
// forces the SIMT token-tile GEMV back, as the A/B arm.
bool verify_uses_small_t() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_VERIFY");
        return value == nullptr || std::string(value) != "tile";
    }();
    return enabled;
}

// A warp takes one whole 128-wide quant group, so K has to hold whole K groups.
inline constexpr std::int32_t kSmallTGroupK = 8 * 128;

// Row block per CTA, i.e. how many mma m-tiles share one activation staging. 16 is the shape the
// kernel was written at and stays the reference; 32 and 48 exist because activations cost more of
// the load pipe than codes do, so widening the block should leave that cost flat while the codes
// grow. Measured on the six verify shapes at tokens=4 that is worth 6-15% on the largest one
// (34816x5120: 84 -> 77 -> 71 us) and is inside noise on the rest -- a net ~0.5-0.9 ms a round,
// which is small enough that the engine-level A/B, not the kernel harness, decides it.
//
// It is NOT the free win the load-mix argument predicts: dropping activation traffic by a third
// only moved the effective rate from 453 to 480 GB/s against a 637 GB/s ceiling, so the kernel is
// not L2-bandwidth-bound after all. Kept env-selectable rather than hardcoded for that reason.
int small_t_rows_per_cta() {
    static const int rows = [] {
        const char* value = std::getenv("NINFER_TERNARY_SMALL_T_ROWS");
        const int parsed  = value == nullptr ? 0 : std::atoi(value);
        return (parsed == 16 || parsed == 32 || parsed == 48) ? parsed : 32;
    }();
    return rows;
}

void launch_small_t(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                    cudaStream_t stream) {
    const std::int32_t rows = w.n;
    const auto* x_ptr       = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes       = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales      = static_cast<const std::uint8_t*>(w.scales);
    auto* out_ptr           = static_cast<__nv_bfloat16*>(out.data);
    const auto grid_for     = [rows](int row_block) {
        return dim3(static_cast<unsigned>(div_up(rows, row_block)), 1u, 1u);
    };
    const auto args = [&](auto tag) {
        constexpr int kRows = decltype(tag)::value;
        ternary_small_t_mma_kernel<8, (kRows == 16 ? 4 : (kRows == 32 ? 3 : 2)), kRows>
            <<<grid_for(kRows), TernarySmallTSchedule::kThreads, 0, stream>>>(
                x_ptr, codes, scales, out_ptr, rows, w.k, x.ne[1], out_row_stride);
    };
    using std::integral_constant;
    switch (small_t_rows_per_cta()) {
    case 16: args(integral_constant<int, 16>{}); break;
    case 48: args(integral_constant<int, 48>{}); break;
    default: args(integral_constant<int, 32>{}); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// The measured winner of the K=64 sweep on the target card: 64 output rows, a 128-token tile, 8
// warps laid out 2x4, two cp.async stages, two CTAs per SM. It beat 64x64, 128x64, 16-warp and
// 32-warp variants (37.0 ms against 41.4-54.8 on the 248320x5120 head at T=1024).
using TernaryMmaPrefillSchedule = TernaryMmaSchedule<64, 128, 64, 32, 32, 2, 2>;

// A token tile narrower than half the block wastes the pipeline, so short verification-shaped T
// stays on the SIMT path even though the kernel would be correct there.
inline constexpr std::int32_t kMmaMinTokens = 64;

template <class Schedule>
void launch_ternary_mma(const Tensor& x, const Weight& w, Tensor& out,
                        std::int32_t out_row_stride, cudaStream_t stream) {
    const std::int32_t rows   = w.n;
    const std::int32_t k      = w.k;
    const std::int32_t tokens = x.ne[1];
    // The reference launcher checks this too (launch_gemm does), and an undersized stride would
    // silently scatter the tile across neighbouring rows rather than fail.
    if (out_row_stride < rows) {
        throw std::invalid_argument("ternary mma: output row stride is smaller than the tile");
    }
    // Restated here rather than left to gemv_admits alone: the kernel stages whole 64-wide K tiles
    // and reads exactly two planes, so a padded K or a high plane would index out of the group.
    // Both call sites pass through gemv_admits today; this is the guard for the next one.
    if (w.padded_shape[1] != k || (k % 128) != 0) {
        throw std::invalid_argument("ternary mma: needs a whole-group K with no padding");
    }
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kBlockRows)),
                    static_cast<unsigned>(div_up(tokens, Schedule::kBlockCols)), 1u);
    const bool full = (rows % Schedule::kBlockRows) == 0 && (tokens % Schedule::kBlockCols) == 0;

    const auto* x_ptr     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes     = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales    = static_cast<const std::uint8_t*>(w.scales);
    auto* out_ptr         = static_cast<__nv_bfloat16*>(out.data);

    if (full) {
        ternary_rowsplit_mma_kernel<Schedule, true><<<grid, Schedule::kThreads, 0, stream>>>(
            x_ptr, codes, scales, out_ptr, rows, k, tokens, w.padded_shape[1], out_row_stride);
    } else {
        ternary_rowsplit_mma_kernel<Schedule, false><<<grid, Schedule::kThreads, 0, stream>>>(
            x_ptr, codes, scales, out_ptr, rows, k, tokens, w.padded_shape[1], out_row_stride);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Wide-token-tile (weight-resident) prefill kernel, ported from the author ternary tree
// (ternary_rowsplit_mma_wide_t.cuh). Replaces the per-8-token weight re-read of small_t with a
// per-64-token window, so one forward pass reads the weights ceil(T/64) times instead of
// ceil(T/8) -- the mechanism behind the author's measured prefill ~2x. Admission mirrors
// mma_wide_admits: PQ2_0, unpadded whole-group K, and no high plane. Selected only when
// NINFER_TERNARY_PREFILL=wide (A/B arm; default stays Mma until engine-side A/B passes).
void launch_ternary_mma_wide(const Tensor& x, const Weight& w, Tensor& out,
                             std::int32_t out_row_stride, cudaStream_t stream) {
    const std::int32_t rows = w.n;
    if (w.padded_shape[1] != w.k || (w.k % kTernaryWideChunkK) != 0 || out_row_stride < rows) {
        throw std::invalid_argument("ternary mma_wide: needs whole-group unpadded K with wide-tile chunk");
    }
    const bool token_grid = ternary_token_grid_enabled();
    const dim3 grid(static_cast<unsigned>(div_up(rows, kTernaryWideRowsPerCta)),
                    token_grid ? static_cast<unsigned>(div_up(x.ne[1], kTernaryWideTokens)) : 1u,
                    1u);
    ternary_wide_t_kernel<false><<<grid, kTernaryWideThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        nullptr, static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), rows, w.k, x.ne[1], out_row_stride, token_grid);
    CUDA_CHECK(cudaGetLastError());
}

// int8 tensor-core prefill rung (#14), ported from the author ternary tree
// (ternary_rowsplit_mma_s8.cuh). Quantizes the activation per token, then runs the s8 mma kernel.
// Requires a caller-provided scratch (see TernaryS8Scratch) because this op runs inside captured
// CUDA graphs, where a lazy cudaMalloc at launch time is illegal. Only PQ2_0 is present in the
// Bonsai artifact, so the PTQ1_0 twin is not wired up here.
//   NINFER_TERNARY_S8=0  -> stay on the bf16 rungs (A/B rollback)
// Admission matches mma_wide_admits (PQ2_0, unpadded whole-group K, no high plane). Selected when
// scratch is present, s8 is enabled, and T >= kTernaryS8MinTokens (33) -- the author's measured
// crossover where s8 wins by 1.20x at T=40, 1.30x at T=64.
void launch_pq2_mma_s8(const Tensor& x, const Weight& w, Tensor& out,
                       std::int32_t out_row_stride, std::int32_t tokens, TernaryS8Scratch scratch,
                       cudaStream_t stream) {
    ternary_s8_quantize_kernel<<<tokens, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), scratch.codes, scratch.scales, w.k);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kTokens = 64;
    constexpr int kWarps  = 4;
    // SCHED3 CHANGE 3: the token axis is grid.y, one CTA per 64-token tile.
    const bool token_grid = ternary_token_grid_enabled();
    const dim3 grid(static_cast<unsigned>(div_up(w.n, TernaryS8Storage<kTokens, kWarps>::kRowsPerCta)),
                    token_grid ? static_cast<unsigned>(div_up(tokens, kTokens)) : 1u, 1u);
    ternary_pq2_mma_s8_kernel<kTokens, kWarps, 3><<<grid, kWarps * 32, 0, stream>>>(
        scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
        w.k, tokens, out_row_stride, nullptr, token_grid);
    CUDA_CHECK(cudaGetLastError());
}

// One call site for the int8 rung. Only PQ2_0 is in the Bonsai artifact, but the dispatcher keeps
// the PTQ1_0 guard so this rung can be extended without changing the call site.
void launch_s8(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
               std::int32_t tokens, TernaryS8Scratch scratch, cudaStream_t stream) {
    if (w.qtype == QType::PTQ1_0_G128) {
        throw std::invalid_argument("ternary s8 mma: PTQ1_0 high plane unsupported in this port");
    }
    launch_pq2_mma_s8(x, w, out, out_row_stride, tokens, scratch, stream);
}
#endif // !NINFER_VOLTA_BUILD

void launch_ternary_gemm_t8(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream,
                            TernaryS8Scratch scratch) {
    (void)scratch; // the int8 rung is tensor-core only and is compiled out under Volta
    // The speculative verify pass runs T = draft + 1 (2..4 here). The reference tiled kernel wastes
    // five of its eight token slots at that size and needs a 128-thread CTA plus seven barriers per
    // output row, which cost more than the whole decode step it was verifying. The small-tile GEMV
    // reads each weight once for all four tokens instead.
    //
    // The row-blocked kernel below was measured on this same path and LOST on Ada: routing T = 2..4
    // to it dropped the MTP decode from 43.0 to 32.8 t/s (en) and 57.7 to 43.9 (zh). NCU says why,
    // and the reason rules the whole family out for the verify shape rather than just that one
    // config. On the 248320-row head the tile kernel already sits at 95.47% achieved occupancy, 36
    // registers and 78% SM throughput -- the ALU pipe ("integer and logic operations") is the top
    // utilizer, 909e6 instructions in 1.70 ms against a 1.32 ms pure-issue floor. It is issue-bound
    // with no occupancy left to buy, so trading registers for fewer loads can only lose: the same
    // head under the row block runs at 63 registers, 65.67% occupancy and 47.95% SM throughput.
    //
    // The verify pass is nonetheless NOT re-reading weights -- nsys shows one forward pass, 401
    // launches, against 407 for a T=1 decode step -- so the cost is per-token issue work, not
    // weight traffic. That is why raising the draft count, which amortises the per-group 2-bit
    // decode over more tokens, is the productive lever here; see the plan document.
    const PrefillRoute route = prefill_route();
    // The fp16 tensor-core route, first because it is the only one that moves the
    // multiply-accumulate off the FP32 pipe. That is the measured ceiling on all three SIMT routes
    // here: each decodes every weight to FP32 and then does one FP32 FMA, which pins prefill at
    // ~38% of the card's FP32 peak, and two structurally unrelated SIMT shapes both land on
    // 111-115 t/s. It refuses shapes it cannot serve -- every T below one 32-token A tile, and every
    // shape whose split-K count exceeds one -- and those fall through to the routes below rather
    // than failing, so this arm is safe to leave selected.
    if (route == PrefillRoute::Mma && ternary_volta_mma_admits(x, w)) {
        launch_ternary_volta_mma(x, w, out, out_row_stride, stream);
        return;
    }
    // The SIMT GEMM is the one route here that stages a K-slice of the activation once per CTA and
    // shares it across kRowsPerCta output rows -- exactly the redundancy the blocked GEMV could not
    // afford at its register budget. NINFER_TERNARY_PREFILL=simt selects it for the T = 2..4 verify
    // band as well, so one build can A/B it against the tile GEMV on both shapes that matter.
    if (route == PrefillRoute::Simt && simt_admits(x, w)) {
        launch_ternary_simt_selected(x, w, out, out_row_stride, stream);
        return;
    }
    if (t2_block_enabled() && gemv_admits(x, w, std::numeric_limits<std::int32_t>::max())) {
        launch_pq2_gemv_tile_block(x, w, out, out_row_stride, stream);
        return;
    }
    // T = 5..16 lives here when the wide tile instantiations are selected; otherwise the dispatcher
    // below hands it to the row-blocked kernel. 16 is the context-lookup width.
    if (tile_wide_enabled() && gemv_admits(x, w, 16)) {
        launch_pq2_gemv_tile(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    if (gemv_admits(x, w, 4)) {
        launch_pq2_gemv_tile(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    // T = 5..8 and every PTQ1_0 / padded-K weight. On Ada this band went to the small-T tensor-core
    // kernel; here the row-blocked GEMV is the only SIMT route that covers it (the tile kernel is
    // instantiated at four tokens), so it is the default rather than a fallback.
    if (route != PrefillRoute::Reference &&
        gemv_admits(x, w, std::numeric_limits<std::int32_t>::max())) {
        launch_pq2_gemv_tile_block(x, w, out, out_row_stride, stream);
        return;
    }
    // Fallback and A/B arm. It still beats the correctness-first reference kernel, which measured
    // 7% of the card's sustained read ceiling where the GEMV shape reaches 66% on the same
    // weights. NINFER_TERNARY_PREFILL=ref forces the reference kernel, so an A/B run can qualify
    // either fast path engine-side (T = 1 alone cannot catch a token-tile or layout error).
    launch_by_qtype<8>(x, w, out, out_row_stride, stream);
}

} // namespace ninfer::ops::detail
