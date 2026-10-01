// ninfer::ops - fused rmsnorm + split fold-rotation launcher: finite dispatch over the one shape
// this port fuses (d = 5120, aligned bf16, fp16 fold). See kernel/ternary_normrotate.cuh for why
// both halves stay bit-identical.
#include "ops/launcher/rmsnorm_rotate.h"

#include "ops/kernel/ternary_normrotate.cuh"
#include "core/device.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

void rmsnorm_rotate_launch(const Tensor& x, const Tensor& norm_weight, float eps, const Tensor& h,
                           Tensor& folded, const float* signs, int n_blk, int rotation_k,
                           cudaStream_t stream) {
    if (x.dtype != DType::BF16 || norm_weight.dtype != DType::BF16 || h.dtype != DType::BF16) {
        throw std::invalid_argument("rmsnorm_rotate: x/weight/h must be BF16");
    }
    if (folded.dtype != DType::FP16) {
        throw std::invalid_argument("rmsnorm_rotate: the decode fold must be FP16");
    }
    if (x.ne[0] != 5120 || norm_weight.ne[0] != x.ne[0] || h.ne[0] != x.ne[0] ||
        h.ne[1] != x.ne[1] || folded.ne[0] != rotation_k || folded.ne[1] != x.ne[1]) {
        throw std::invalid_argument("rmsnorm_rotate: shape mismatch on d/rows/tokens");
    }
    if (signs == nullptr || n_blk <= 0) {
        throw std::invalid_argument("rmsnorm_rotate: the folded basis needs a sign block");
    }
    const auto x_addr  = reinterpret_cast<std::uintptr_t>(x.data);
    const auto w_addr  = reinterpret_cast<std::uintptr_t>(norm_weight.data);
    const auto h_addr  = reinterpret_cast<std::uintptr_t>(h.data);
    const auto o_addr  = reinterpret_cast<std::uintptr_t>(folded.data);
    const bool aligned2 =
        ((x_addr | w_addr | h_addr | o_addr) & (alignof(__nv_bfloat162) - 1)) == 0;
    if (!aligned2) { throw std::invalid_argument("rmsnorm_rotate: addresses must be 2-aligned"); }

    const std::int32_t tokens = x.ne[1];
    // NINFER_TERNARY_NORMROT=2 launches the 640-thread shape: phase 1 still computes as the
    // 256-thread norm (idle threads are inert), phase 2 gets 20 warps = five four-warp groups
    // so all five D1024 units run in ONE round -- the parallelism the 256-thread shape lost.
    // =1 (or unset) keeps the original 256-thread launch.
    static const int mode = [] {
        const char* env = std::getenv("NINFER_TERNARY_NORMROT");
        return (env != nullptr && std::atoi(env) == 2) ? 2 : 1;
    }();
    const auto* x2   = reinterpret_cast<const __nv_bfloat162*>(x.data);
    const auto* w2   = reinterpret_cast<const __nv_bfloat162*>(norm_weight.data);
    auto* h2         = reinterpret_cast<__nv_bfloat162*>(h.data);
    if (mode == 2) {
        rmsnorm_rotate_kernel<RmsEpilogue::Offset, 640, 10, true, 5120, true>
            <<<static_cast<unsigned int>(tokens), 640, 0, stream>>>(
                x2, w2, nullptr, h2, folded.data, signs, n_blk, rotation_k, tokens, eps);
    } else {
        rmsnorm_rotate_kernel<RmsEpilogue::Offset, 256, 10, true, 5120, true>
            <<<static_cast<unsigned int>(tokens), 256, 0, stream>>>(
                x2, w2, nullptr, h2, folded.data, signs, n_blk, rotation_k, tokens, eps);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
