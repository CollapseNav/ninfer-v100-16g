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

#include <cstdint>
#include <cstdlib>

namespace ninfer::runtime {

[[nodiscard]] inline bool kv_ring_requested() noexcept {
    const char* text = std::getenv("NINFER_KV_WINDOW");
    if (text == nullptr || text[0] == '\0') { return false; }
    return std::strtol(text, nullptr, 10) > 0;
}

// The head of the context that the ring may NEVER demote: the attention sink.
//
// MEASURED FAILURE THIS ADDRESSES. A ~37.5k context served with a 16,384-token window, asked
// "继续" (an empty continuation query with no lexical overlap with anything), degenerated into
// 400 tokens of the single phrase `"继续。" and ` repeated 76 times. The SAME prompt with the
// context fully resident produced coherent reasoning. The cause is not the window size by itself:
// retention is lexical, so a query with no signal retains nothing, and once the first tokens --
// the task statement -- are demoted too, the model has no anchor left at all and collapses.
// With a lexically RICH query the same window recalled a needle from the start of the prompt, so
// this is specifically the lexically-empty case.
//
// StreamingLLM reaches the same conclusion for sliding windows: keeping a few initial tokens as
// attention sinks is what stops the model from collapsing once the context outgrows the window.
// Four pages is 256 tokens: enough to hold a task statement, and ~8 MiB of a ~1.7 GiB pool.
// NINFER_KV_SINK_TOKENS overrides it; 0 restores the previous behaviour (spill the head too).
[[nodiscard]] inline std::uint32_t kv_ring_sink_tokens() noexcept {
    constexpr std::uint32_t kDefaultSinkTokens = 256;
    const char* text = std::getenv("NINFER_KV_SINK_TOKENS");
    if (text == nullptr || text[0] == '\0') { return kDefaultSinkTokens; }
    const long value = std::strtol(text, nullptr, 10);
    return value > 0 ? static_cast<std::uint32_t>(value) : 0U;
}

} // namespace ninfer::runtime
