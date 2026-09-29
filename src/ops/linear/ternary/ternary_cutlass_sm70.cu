// See ternary_cutlass_sm70.h for why this route exists and what it trades.
#include "ops/linear/ternary/ternary_cutlass_sm70.h"

#include "core/device.h"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {

using ElementAccumulator     = float;
using ElementComputeEpilogue = float;
using ElementInput           = cutlass::half_t;
using ElementOutput          = cutlass::bfloat16_t;

// The sm70 fp16 tensor-op GEMM from fp8_cutlass_sm70.cu. The standalone probe measured the
// 128x128x32 tile at 94-98 TFLOP/s on this model's real shapes -- but only with the A panel resident
// in L2. In situ that panel is evicted by the fp16 weight stream, and A is re-read (N/BN) times, so
// BN is the lever that matters: at BN=128 the whole-model A traffic is ~1.1 TB against ~0.55 TB at
// BN=256, on top of ~1.1 TB of unavoidable B traffic. NINFER_TERNARY_CUTLASS_TILE picks between
// them at run time so one build can A/B the traffic model.
//
// kStages is not a parameter here, and there is no env knob for it. CUTLASS's sm70 tensor-op path
// has exactly one kernel: `kernel::DefaultGemm<... arch::Sm70 ...>` is specialised only for
// Stages == 2, and the threadblock kernel it selects is MmaPipelined, which asserts
// `kStages == 2` (mma_pipelined.h:137). Asking device::Gemm for Stages 3 or 4 leaves
// kernel::DefaultGemm an incomplete type and the compile dies inside cutlass/gemm/device/gemm.h.
// Measured: 55 errors, all downstream of that one missing specialisation. Deeper software
// pipelining on Volta is therefore not a tuning knob but a hand-written kernel, and the sm70
// MmaMultistage path does not exist either (the multistage DefaultMma specialisations are all
// keyed on Sm80/Sm75, and the sm70 shared-memory iterators have no stage dimension to index).
template <int BN>
using GemmT = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor, ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<128, BN, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
        ElementComputeEpilogue>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

template <int BN>
std::size_t gemm_workspace_bytes_t(int n_chunk, int k, int cols) {
    const cutlass::gemm::GemmCoord shape(cols, n_chunk, k);
    typename GemmT<BN>::Arguments args{shape, {nullptr, k}, {nullptr, k}, {nullptr, n_chunk},
                                       {nullptr, n_chunk}, {1.0F, 0.0F}, 1};
    return GemmT<BN>::get_workspace_size(args);
}

// The dequantised weight chunk is capped by bytes, not rows, so the arena reservation does not
// scale with K. 64 MiB holds ~6500 rows at k=5120 and ~1800 at k=17408.
constexpr std::size_t kMaxChunkBytes = 64u * 1024u * 1024u;
constexpr int kMinChunkRows          = 512;
// CUTLASS's C operand advances by a whole chunk, so the chunk length has to keep `c + n0` on a
// 128-bit boundary for bf16: 8 elements. 64 is far past that and keeps the arithmetic tidy.
constexpr int kChunkAlign = 64;

// Below this token count the GEMM is a GEMV and CUTLASS is the wrong tool: the probe measured
// 0.3 TFLOP/s at M=1 and 4.8 at M=16, against the fused kernel's 36.7 at long T.
constexpr std::int32_t kMinTokens = 256;

// The chunk length is a function of K alone, never of N. Two of the four planners that have to
// reserve this buffer -- the attention and GDN parents -- are handed only (input_rows, max_tokens)
// and do not know the output row counts at all, so an N-dependent reservation could not be stated
// there. Making it K-only costs at most one wasted chunk on a weight narrower than the cap, and the
// launcher shrinks to the real N when it allocates.
int chunk_rows_for(std::int32_t k) {
    const std::size_t per_row = 2u * static_cast<std::size_t>(k);
    const std::size_t cap = std::max<std::size_t>(kMaxChunkBytes / std::max<std::size_t>(per_row, 1u),
                                                  kMinChunkRows);
    return static_cast<int>(std::max<std::size_t>(cap / kChunkAlign, 1u) * kChunkAlign);
}

int aligned_rows_for(std::int32_t n, std::int32_t k) {
    const int cap = chunk_rows_for(k);
    const std::size_t rows = std::min<std::size_t>(static_cast<std::size_t>(n),
                                                   static_cast<std::size_t>(cap));
    return static_cast<int>(std::max<std::size_t>(rows / kChunkAlign, 1u) * kChunkAlign);
}

// Bit pattern of a __half2, without taking the address of an array.
__forceinline__ __device__ unsigned bits_of(__half2 h) {
    return *reinterpret_cast<unsigned*>(&h);
}

// PQ2_0_G128 -> fp16 for rows [row0, row0 + rows).
//
// One thread owns two code bytes -- eight consecutive weights of one group -- and writes them as a
// single 16-byte vector. Sixteen threads therefore cover one group's 32 code bytes and its 256 bytes
// of output, and a warp's stores are one contiguous 512-byte run.
//
// Both halves of that mapping were measured, in dq_probe.cu:
//   * the obvious one-byte-per-thread form needed two 64-bit divisions by a runtime `groups` just to
//     recover (row, group, byte), which put it at ~400 GB/s of the card's 900;
//   * a 4-byte-per-thread form with two 16-byte stores put a warp's first store at byte stride 32,
//     so every 32-byte sector was half-written and re-fetched: its store pattern alone measured
//     367 GB/s against 1112 GB/s for this one.
//
// The value is (code - 1) * scale with code in {0,1,2}. That product is exactly representable in
// fp16 (the multiplier is -1, 0, 1 or 2), so computing it in fp32 and rounding once reproduces the
// fused kernel's fp16 magic-number decode bit for bit -- the fused route's __hmul2 of an exact
// (code-1) by an fp16 scale is the same real number, and the run below is token-identical to the
// FP32 reference on the strength of it.
// kMode 0 is the real kernel. 1 and 2 are in-situ TIMING PROBES that bisect it -- the standalone
// microbenchmark already showed the store pattern was worth 3x, so if the in-situ figure is still
// far below that, one of these two sides has to own the difference. Both produce wrong output by
// construction and are never a default.
//   1 kDqWriteOnly -- keep the store, replace the code word with a constant
//   2 kDqReadOnly  -- keep every load, make the store unreachable
enum TernaryDqMode { kDqFull = 0, kDqWriteOnly = 1, kDqReadOnly = 2 };

template <int kMode>
__global__ void dequant_pq2_chunk_to_fp16(const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ scales,
                                          std::int32_t row0, std::int32_t rows, std::int32_t k,
                                          std::int32_t groups,
                                          cutlass::half_t* __restrict__ out) {
    constexpr int kGroupK   = PQ2RowSplitStorage::kGroupK;
    constexpr int kCodeB    = PQ2RowSplitStorage::kCodeBytesPerGroup;
    constexpr int kScaleB   = PQ2RowSplitStorage::kScaleBytesPerGroup;
    constexpr int kPerByte  = PQ2RowSplitStorage::kCodesPerByte;
    constexpr int kPerWord  = 2;                              // code bytes per thread
    constexpr int kWeights  = kPerWord * kPerByte;            // 8
    constexpr int kThreadsPerGroup = kCodeB / kPerWord;       // 16

    const std::int32_t g = static_cast<std::int32_t>(blockIdx.x) * (256 / kThreadsPerGroup) +
                           static_cast<std::int32_t>(threadIdx.x) / kThreadsPerGroup;
    const std::int32_t r = static_cast<std::int32_t>(blockIdx.y);
    if (r >= rows || g >= groups) { return; }
    const std::int32_t bq = (static_cast<std::int32_t>(threadIdx.x) % kThreadsPerGroup) * kPerWord;

    // One 16-bit code word per thread; codes with the same scale, so the 8 weights it holds are
    // exactly the 8 halves of the 16-byte store below.
    const std::uint16_t word =
        kMode == kDqWriteOnly
            ? static_cast<std::uint16_t>(0x5555u)
            : *reinterpret_cast<const std::uint16_t*>(
                  codes + static_cast<std::int64_t>(row0 + r) * groups * kCodeB +
                  static_cast<std::int64_t>(g) * kCodeB + bq);
    const __half scale = __ushort_as_half(
        reinterpret_cast<const std::uint16_t*>(scales + static_cast<std::int64_t>(row0 + r) *
                                                             groups * kScaleB)[g]);
    const float s = __half2float(scale);

    __half2 p[kWeights / 2];
#pragma unroll
    for (int i = 0; i < kWeights / 2; ++i) {
        const int c0 = static_cast<int>((word >> (4 * i)) & 0x3u);
        const int c1 = static_cast<int>((word >> (4 * i + 2)) & 0x3u);
        p[i] = __floats2half2_rn(static_cast<float>(c0 - 1) * s, static_cast<float>(c1 - 1) * s);
    }
    // k is a multiple of 16 and the group base is a multiple of 128, so this is 16-byte aligned.
    cutlass::half_t* dst = out + static_cast<std::int64_t>(r) * k +
                           static_cast<std::int64_t>(g) * kGroupK + bq * kPerByte;
    uint4 v;
    v.x = bits_of(p[0]);
    v.y = bits_of(p[1]);
    v.z = bits_of(p[2]);
    v.w = bits_of(p[3]);
    if (kMode == kDqReadOnly) {
        // Unreachable in practice, but the compiler cannot prove it, so every load above stays live.
        if (v.x == 0xdeadbeefu && v.y == 0xdeadbeefu) { *reinterpret_cast<uint4*>(dst) = v; }
    } else {
        *reinterpret_cast<uint4*>(dst) = v;
    }
}

// bf16 -> fp16 into a separate fp16 buffer. It is deliberately NOT in place: the activation can be
// shared by several folded weights (the attention and GDN parents feed four and three), so writing
// fp16 over the caller's bf16 buffer would corrupt the other projections.
__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    cutlass::half_t* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = cutlass::half_t(__bfloat162float(in[i]));
}

// The audit counters that located the prefill-chunk repetition (a host entry count plus a device
// block counter, which is what separated "the engine launches the model per chunk" from "a captured
// graph is replayed") have been removed. They cost an atomicAdd per dequant block on the default
// path -- roughly 20k blocks x 1300 launches per prefill -- and the finding is recorded in
// docs/ternary-port.md instead. The probe modes below are kept: they are env-gated, off by default,
// and they are how every number in that document was obtained.

bool route_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_CUTLASS");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

std::int32_t min_tokens() {
    static const std::int32_t value = [] {
        const char* env = std::getenv("NINFER_TERNARY_CUTLASS_MIN_T");
        return env == nullptr ? kMinTokens : std::max(1, std::atoi(env));
    }();
    return value;
}

// 0 none, 1 dequant only (skip the GEMM), 2 GEMM only (skip the dequant), 3 dequant TWICE and skip
// the GEMM. Modes 1-3 are timing probes and produce wrong output by construction, exactly like
// NINFER_TERNARY_MMA_PROBE. Mode 3 exists because subtracting two probe runs to size the dequant
// gave 0.55 s, while the kernel measures 46 GB at 413 GB/s standalone -- i.e. 0.11 s. The delta
// between mode 1 and mode 3 is one whole dequant pass with every other term of the model held fixed,
// so it cannot be contaminated by what removing the GEMM does to the rest of the schedule.
int probe_mode() {
    static const int value = [] {
        const char* env = std::getenv("NINFER_TERNARY_CUTLASS_PROBE");
        if (env == nullptr) { return 0; }
        const std::string text(env);
        if (text == "dequant") { return 1; }
        if (text == "gemm") { return 2; }
        if (text == "dequant2") { return 3; }
        if (text == "dqwrite") { return 4; }
        if (text == "dqread") { return 5; }
        return 0;
    }();
    return value;
}

bool tile_256() {
    static const bool value = [] {
        const char* env = std::getenv("NINFER_TERNARY_CUTLASS_TILE");
        return env != nullptr && std::string(env) == "128x256";
    }();
    return value;
}

template <int BN>
void run_chunks(const Tensor& x_folded, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                WorkspaceArena& ws, cudaStream_t stream, int rows, const int probe) {
    constexpr int kGroupK = PQ2RowSplitStorage::kGroupK;
    const std::int32_t n = w.n;
    const std::int32_t k = w.k;
    const std::int32_t t = x_folded.ne[1];
    const std::int32_t groups = k / kGroupK;

    const std::size_t gemm_bytes = gemm_workspace_bytes_t<BN>(rows, k, t);
    auto scope = ws.scope();
    const DeviceSpan weight_span = ws.alloc_bytes(static_cast<std::size_t>(rows) * k * 2u);
    const DeviceSpan input_span  = ws.alloc_bytes(static_cast<std::size_t>(k) * t * 2u);
    const DeviceSpan gemm_span   = ws.alloc_bytes(gemm_bytes + 256u);
    auto* weight = static_cast<cutlass::half_t*>(weight_span.data);
    auto* input  = static_cast<cutlass::half_t*>(input_span.data);

    if (probe != 2) {
        const std::int64_t count = static_cast<std::int64_t>(k) * t;
        bf16_to_fp16_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x_folded.data), input, count);
        CUDA_CHECK(cudaGetLastError());
    }
    auto* a = probe == 2 ? reinterpret_cast<const ElementInput*>(x_folded.data) : input;
    auto* c = static_cast<ElementOutput*>(out.data);

    for (std::int32_t n0 = 0; n0 < n; n0 += rows) {
        const int chunk = static_cast<int>(std::min<std::int32_t>(rows, n - n0));
        if (probe != 2) {
            constexpr int kThreadsPerGroup = PQ2RowSplitStorage::kCodeBytesPerGroup / 2;
            constexpr int kThreads         = 256;
            const int groups_per_block = kThreads / kThreadsPerGroup;
            const dim3 dq_grid(static_cast<unsigned>((groups + groups_per_block - 1) / groups_per_block),
                               static_cast<unsigned>(chunk), 1u);
            // mode 1 and mode 3 differ only in how many times this runs; 4 and 5 bisect the kernel.
            const int passes = probe == 3 ? 2 : 1;
            for (int pass = 0; pass < passes; ++pass) {
                if (probe == 4) {
                    dequant_pq2_chunk_to_fp16<kDqWriteOnly><<<dq_grid, kThreads, 0, stream>>>(
                        static_cast<const std::uint8_t*>(w.qdata),
                        static_cast<const std::uint8_t*>(w.scales), n0, chunk, k, groups, weight);
                } else if (probe == 5) {
                    dequant_pq2_chunk_to_fp16<kDqReadOnly><<<dq_grid, kThreads, 0, stream>>>(
                        static_cast<const std::uint8_t*>(w.qdata),
                        static_cast<const std::uint8_t*>(w.scales), n0, chunk, k, groups, weight);
                } else {
                    dequant_pq2_chunk_to_fp16<kDqFull><<<dq_grid, kThreads, 0, stream>>>(
                        static_cast<const std::uint8_t*>(w.qdata),
                        static_cast<const std::uint8_t*>(w.scales), n0, chunk, k, groups, weight);
                }
                CUDA_CHECK(cudaGetLastError());
            }
        }
        if (probe == 1 || probe == 3 || probe == 4 || probe == 5) { continue; }

        const cutlass::gemm::GemmCoord shape(t, chunk, k);
        typename GemmT<BN>::Arguments args{shape, {a, k}, {weight, k}, {c + n0, out_row_stride},
                                           {c + n0, out_row_stride}, {1.0F, 0.0F}, 1};
        GemmT<BN> op;
        if (op.can_implement(args) != cutlass::Status::kSuccess) {
            throw std::runtime_error(
                "ternary cutlass: CUTLASS can_implement failed [T=" + std::to_string(t) + ", N=" +
                std::to_string(chunk) + " at " + std::to_string(n0) + ", K=" + std::to_string(k) +
                ", ldc=" + std::to_string(out_row_stride) + ", A%" +
                std::to_string(reinterpret_cast<std::uintptr_t>(a) % 16) + ", B%" +
                std::to_string(reinterpret_cast<std::uintptr_t>(weight) % 16) + ", C%" +
                std::to_string(reinterpret_cast<std::uintptr_t>(c + n0) % 16) + "]");
        }
        if (op.initialize(args, gemm_span.data, stream) != cutlass::Status::kSuccess) {
            throw std::runtime_error("ternary cutlass: CUTLASS initialize failed");
        }
        if (op(stream) != cutlass::Status::kSuccess) {
            throw std::runtime_error("ternary cutlass: CUTLASS gemm failed");
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

bool ternary_cutlass_sm70_admits(const Weight& w, std::int32_t tokens) noexcept {
    if (!route_enabled()) { return false; }
    constexpr int kGroupK = PQ2RowSplitStorage::kGroupK;
    if (w.qtype != QType::PQ2_0_G128) { return false; }
    if (w.qhigh != nullptr || w.padded_shape[1] != w.k) { return false; }
    if (w.n <= 0 || w.k <= 0 || tokens < min_tokens()) { return false; }
    if (w.k % kGroupK != 0 || w.k % 32 != 0) { return false; }
    return true;
}

std::size_t ternary_cutlass_sm70_workspace_bytes(std::int32_t k, std::int32_t cols) {
    if (!route_enabled() || k <= 0 || cols < min_tokens()) { return 0; }
    constexpr int kGroupK = PQ2RowSplitStorage::kGroupK;
    if (k % kGroupK != 0 || k % 32 != 0) { return 0; }
    // The reserved chunk is the K-only cap, not what the launcher will actually allocate: any
    // shape-specific shrink is a strict subset, so over-reserving is the safe direction and the
    // launcher can never ask the arena for more than the planner declared.
    const int rows = chunk_rows_for(k);
    const std::size_t gemm = tile_256() ? gemm_workspace_bytes_t<256>(rows, k, cols)
                                        : gemm_workspace_bytes_t<128>(rows, k, cols);
    return static_cast<std::size_t>(rows) * k * 2u + static_cast<std::size_t>(k) * cols * 2u +
           gemm + 512u;
}

void ternary_cutlass_sm70_launch(const Tensor& x_folded, const Weight& w, Tensor& out,
                                 std::int32_t out_row_stride, WorkspaceArena& ws,
                                 cudaStream_t stream) {
    constexpr int kGroupK = PQ2RowSplitStorage::kGroupK;
    if (w.k % kGroupK != 0) {
        throw std::invalid_argument("ternary cutlass: K is not a whole number of groups");
    }
    if (out_row_stride < w.n) {
        throw std::invalid_argument("ternary cutlass: output row stride is smaller than the tile");
    }
    const int rows  = aligned_rows_for(w.n, w.k);
    if (rows <= 0) { throw std::invalid_argument("ternary cutlass: N is below one aligned chunk"); }
    const int probe = probe_mode();
    if (tile_256()) {
        run_chunks<256>(x_folded, w, out, out_row_stride, ws, stream, rows, probe);
    } else {
        run_chunks<128>(x_folded, w, out, out_row_stride, ws, stream, rows, probe);
    }
}

#else

bool ternary_cutlass_sm70_admits(const Weight&, std::int32_t) noexcept { return false; }

std::size_t ternary_cutlass_sm70_workspace_bytes(std::int32_t, std::int32_t) { return 0; }

void ternary_cutlass_sm70_launch(const Tensor&, const Weight&, Tensor&, std::int32_t,
                                 WorkspaceArena&, cudaStream_t) {
    throw std::invalid_argument("ternary cutlass: this build has no Volta tensor-core route");
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
