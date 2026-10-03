#pragma once

// ninfer::ops - a projection whose weights are GGUF block matrices assembled from PARTS.

#include "core/tensor.h"

#include <cstdint>
#include <vector>

namespace ninfer::ops {

/**
 * One stored GGUF matrix and where its rows land: `weight` produces rows
 * [row, row + weight.n) of output tensor `output`, whose index is the overload's own output order
 * (for attn_input_proj: 0 = q, 1 = gate, 2 = k, 3 = v).
 *
 * A fused parent whose parts share one stored object is expressed as several parts over ROW-SLICE
 * views of that object rather than as one part, because one GgufProjectionPart targets exactly one
 * output tensor. A row slice is a pointer and row-count adjustment: a GGUF row is always a whole
 * number of blocks, so rows are byte-uniform and no repacking is involved.
 */
struct GgufProjectionPart {
    Weight weight;
    std::int32_t output = 0;
    std::int32_t row    = 0;
};

/**
 * A projection over GGUF parts.
 *
 * This overload exists because a GGUF projection's parts may differ in FORMAT and therefore cannot
 * be fused into one stored tensor: the ModelScope Swift-1.5 Qwen3.8-27B artifacts store attention
 * gate as gguf_iq2_xxs and value as gguf_iq2_s, and the MLP gate/up pair likewise. Every part is
 * projected by the vendored ggml kernels in one gguf_project call, which quantizes the activation to
 * ggml's q8_1 once and reuses it for every part.
 */
struct GgufProjectionWeights {
    std::vector<GgufProjectionPart> parts;
};

} // namespace ninfer::ops
