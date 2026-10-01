// ninfer::ops - residual_add wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/residual_add.h"

#include "ops/launcher/residual_add.h" // detail::residual_add_launch

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t) {
    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument("residual_add: y/x dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { return 0; }
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("residual_add: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

} // namespace

void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream) {
    if (y.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("residual_add: y/x must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (y.ne[d] != x.ne[d]) {
            throw std::invalid_argument("residual_add: y/x shapes must match");
        }
    }
    if (numel_allow_zero(x) == 0) { return; }
    if (!y.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("residual_add: y/x must be contiguous");
    }
    if (y.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("residual_add: y/x data must be non-null");
    }


    // PROBE (NINFER_TERNARY_PROBE_SKIP_RESIDUAL): do not launch this kernel at all. Sizes and every
    // downstream address are unchanged -- the consumer reads whatever the buffer already holds --
    // so the wall-clock delta against a control run prices launch + loads + stores + ALU of this
    // one kernel. Numerically wrong by construction; never a default.
    static const bool probe_skip = [] {
        const char* env = std::getenv("NINFER_TERNARY_PROBE_SKIP_RESIDUAL");
        return env != nullptr && std::string(env) != "0";
    }();
    if (probe_skip) { return; }
    detail::residual_add_launch(y, x, stream); // single variant -> direct dispatch
}

} // namespace ninfer::ops
