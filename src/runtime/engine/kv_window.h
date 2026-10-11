#pragma once

// KVMem ring switch: lets --kv-capacity (the DEVICE pool) be SMALLER than --max-context (the
// LOGICAL context), by demoting the pages that do not fit to the Host KV tier.
//
// WHY THIS IS ONE DEFINITION AND NOT TWO
//   The CLI gate (serve/serve_options.cpp) and the target gates
//   (targets/qwen3_6/impl/runtime/layouts_impl.h, targets/qwen3_6/impl/state/decoder_state.cpp)
//   must read the SAME variable, or the process passes one gate and dies at the other. That exact
//   split is a shipped defect in the reference project this port follows (ninfer-fusion-kvmem,
//   recorded as "B30"): the CLI read NINFER_KV_RING while the engine read NINFER_KV_WINDOW, so an
//   operator following the engine's own printed advice set a variable that did nothing, passed the
//   CLI, and then died at startup with "kv_capacity must be at least max_context" -- never serving
//   a request. Every gate calls kv_ring_requested() so the predicate cannot drift again.
//
// SEMANTICS
//   NINFER_KV_WINDOW=<tokens>, integer > 0  -> ring ON; the value is the resident window in tokens.
//   Unset, empty, or <= 0                   -> ring OFF, and every gate is unchanged from the dense
//                                             configuration.
//
// SCOPE OF THIS STEP (stage 1, step 1)
//   This only DECOUPLES the two capacities so a small pool is configurable and the layout can be
//   built. It does not yet demote or retrieve anything: with the ring on and a pool below the
//   logical context, activation still reserves the full entitlement and will fail at
//   DeviceKVPagePool::resize_reservation (core/paged_kv_cache.cpp). That failure is the next
//   step's work, not a regression -- with the ring OFF nothing here is reachable.

#include <cstdlib>

namespace ninfer::runtime {

[[nodiscard]] inline bool kv_ring_requested() noexcept {
    const char* text = std::getenv("NINFER_KV_WINDOW");
    if (text == nullptr || text[0] == '\0') { return false; }
    return std::strtol(text, nullptr, 10) > 0;
}

} // namespace ninfer::runtime
