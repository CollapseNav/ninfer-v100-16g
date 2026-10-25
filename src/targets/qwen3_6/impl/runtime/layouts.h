#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "runtime/engine/kv_window.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/state_image.h>
#include <ninfer/targets/qwen3_6/startup_features.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

// Pages ONE KVMem ring lane may hold resident on the Device: its resident window, one prefill chunk
// (a chunk is mapped before any of it can be demoted, so the peak is window + chunk), and the
// attention sink, which the ring never demotes. This is the bound the ring actually honours, so the
// pool minimum, the plan's Device expectation and the owner footprint all use it -- with
// --max-concurrency N, `N * ring_lane_resident_pages(...) + slack` is what the Device pool needs.
[[nodiscard]] inline std::uint32_t ring_lane_resident_pages(std::uint32_t window_pages,
                                                            std::uint32_t chunk_pages) {
    const auto page_tokens = static_cast<std::uint32_t>(kPagedKVPageSize);
    const std::uint32_t sink_pages = (runtime::kv_ring_sink_tokens() + page_tokens - 1U) /
                                     page_tokens;
    return window_pages + chunk_pages + sink_pages;
}

// Device pages ONE ring lane may hold. One lane gets the whole pool, exactly as before multi-lane
// support existed; N lanes each get an equal share. minimum_main_page_groups() guarantees that
// share covers ring_lane_resident_pages(), so a lane can always map a whole chunk before it has to
// demote anything.
[[nodiscard]] inline std::uint32_t ring_lane_budget_pages(std::uint32_t pool_capacity_pages,
                                                          std::uint32_t lanes) {
    if (lanes <= 1) { return pool_capacity_pages; }
    return std::max<std::uint32_t>(pool_capacity_pages / lanes, 1U);
}

using TensorLayout                              = TensorRegion;
inline constexpr std::uint32_t kCausalScoreTile = 1024;

struct DFlashPersistentLayout {
    std::optional<qwen3_6::PagedKVCacheLayout> full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return full ? full->payload_bytes() : 0;
    }
};

struct PersistentLayout {
    qwen3_6::DecoderStateLayout decoder;
    qwen3_6::StateImageDeviceLayout state_images;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<GdnReplayRecordLayout> mtp_lookup_replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3_6::RoundStateLayout round;
    TensorLayout prefill_hidden;
    std::optional<TensorLayout> score_hidden;
    TensorLayout token_counts;
    TensorLayout sampling_config;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

struct VisionWorkspacePlan {
    std::uint32_t max_merged_tokens    = 0;
    std::size_t general_capacity_bytes = 0;
    std::size_t encode_peak_bytes      = 0;
    std::size_t handoff_offset_bytes   = 0;
    std::size_t handoff_capacity_bytes = 0;
    std::size_t capacity_bytes         = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill     = 0;
    std::size_t ordinary_round   = 0;
    std::size_t mtp_prefill      = 0;
    std::size_t mtp_round        = 0;
    std::size_t dflash_context   = 0;
    std::size_t dflash_round     = 0;
    std::size_t causal_score     = 0;
    std::size_t general_capacity = 0;
    std::optional<VisionWorkspacePlan> vision;
    std::size_t capacity = 0;
};

struct SequencePlanningInputs {
    WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    KvCacheStorage kv_storage              = KvCacheStorage::BFloat16;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    bool use_cuda_graph = true;
    bool causal_scoring = false;
    int device          = 0;
    ContextCacheOptions context_cache;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct SequencePlanImpl<NINFER_QWEN36_VARIANT> {
    typename NINFER_QWEN36_VARIANT::WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t kv_capacity              = 0;
    std::uint32_t main_page_groups         = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    KvCacheStorage kv_storage              = KvCacheStorage::BFloat16;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    bool use_cuda_graph = true;
    bool causal_scoring = false;
    int device          = 0;
    ContextCacheOptions context_cache;
    NINFER_QWEN36_RUNTIME_NS::PersistentLayout persistent;
    NINFER_QWEN36_RUNTIME_NS::WorkspacePlan workspace;
    std::size_t graph_allowance_bytes    = 0;
    std::size_t device_reservation_bytes = 0;
};

template <>
struct SequencePlannerImpl<NINFER_QWEN36_VARIANT> {
    NINFER_QWEN36_RUNTIME_NS::SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl<NINFER_QWEN36_VARIANT>> minimum;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using SequencePlanImpl = qwen3_6::detail::SequencePlanImpl<Variant>;

[[nodiscard]] std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
