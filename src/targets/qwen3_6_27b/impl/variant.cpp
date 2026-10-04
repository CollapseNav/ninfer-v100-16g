#include "targets/qwen3_6_27b/impl/variant.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/linear/gguf/gguf_linear.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

#ifdef NINFER_VOLTA_BUILD
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::A16Only;
constexpr ops::LinearPolicy kFp8TextPolicy   = ops::LinearPolicy::A16Only;
#else
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::AllowA4;
constexpr ops::LinearPolicy kFp8TextPolicy   = ops::LinearPolicy::AllowA8;
#endif

ops::LinearPolicy text_policy(const Weight& weight) {
    switch (weight.qtype) {
    case QType::NVFP4:
        return kNvfp4TextPolicy;
    case QType::FP8_E4M3FN_ROW_BF16S:
        return kFp8TextPolicy;
    default:
        return ops::LinearPolicy::A16Only;
    }
}

constexpr std::size_t kMinimumLeafWorkspaceBytes = 1;

// The GDN input-projection leaves below need this: the GGUF parts route runs a ggml projection INSIDE
// the leaf, so the leaf has to hold its transient too. (It is defined here rather than next to its
// other users because those leaves are the first thing in the file that needs it.)
//
// The GGUF parts route quantizes the activation to ggml's q8_1 once per projection and reuses it for
// every part, so its transient bytes depend on the part shapes. At plan time only the weights profile
// is known -- the plan is queried before any weight is bound -- so size for the FUSED parent width the
// other profiles use. That is an over-estimate for every layer, which is safe because the arena is
// sized once for the whole model and no part is ever wider than its parent.
// The GGUF parts route quantizes the activation to ggml's q8_1 once per DISTINCT input gather, and
// gguf_project_workspace_bytes sizes that per shape it is handed. This port's profile-level cases run
// before any weight is bound, so they cannot know the real part list -- and passing one shape
// under-counted the activations by the part count, which is four for attention (q|gate|k|v) and four
// for GDN (q|k|v|z). That shortfall is what surfaced as std::bad_alloc at context >= 384. Sizing for
// the widest part list any layer can have is safe: the arena is sized once for the whole model and
// over-reserving a few activation buffers is the trade the tree already makes elsewhere.
std::size_t gguf_projection_bytes(std::int32_t rows, std::int32_t first, std::int32_t last,
                                  std::int32_t parts = 4) {
    const ops::detail::GgufShape parent{QType::GGUF_IQ2_XXS, rows, TextConfig::hidden};
    const std::vector<ops::detail::GgufShape> shapes(static_cast<std::size_t>(parts), parent);
    return ops::detail::gguf_project_workspace_bytes(shapes, first, last);
}

// Folded (rotated-basis) ternary activation scratch: a [input_width, last] BF16 buffer that
// ternary weights need before their matmul (P, then the signs, then the nominal Hadamard).
//
// The folded ternary port deliberately keeps the groupwise-int identity -- the binding layer
// resolves each weight's format from the artifact's own declaration, so the weights profile only
// decides which tensors exist, not how they are encoded -- which means this profile is shared by
// both artifacts and the reservation cannot be made conditional on it. The cost is one
// activation-sized buffer per ternary op, the same order as the activation roots the stage
// already reserves, and it is what lets the ternary artifact construct its CUDA graphs at all.
std::size_t folded_rotation_bytes(std::int32_t input_width, std::int32_t last) {
    return ops::linear_workspace_capacity_bytes(QType::PQ2_0_G128, input_width, input_width,
                                                ops::LinearPolicy::A16Only, 1, last);
}

// NINFER_GGUF_DUMP_CONV=<path> appends the conv state this leaf produced, one record per call, and a
// text log of the first values it was given. It only produces valid readings with --no-cuda-graph:
// during capture the kernels do not run and the host side is visited once, so a probe inside a leaf
// observes nothing about a replayed decode. The stream is synchronized first because a plain
// cudaMemcpy runs on the legacy default stream and would read ahead of the kernels.
void dump_conv_state(const Tensor& conv_states, const Tensor& projected, cudaStream_t stream) {
    static const char* path = std::getenv("NINFER_GGUF_DUMP_CONV");
    if (path == nullptr) { return; }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { return; }
    static std::FILE* file = std::fopen(path, "wb");
    if (file != nullptr && conv_states.data != nullptr && conv_states.dtype == DType::BF16) {
        const std::size_t bytes =
            static_cast<std::size_t>(conv_states.numel()) * sizeof(__nv_bfloat16);
        std::vector<std::byte> host(bytes);
        if (cudaMemcpy(host.data(), conv_states.data, bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess) {
            std::fwrite(host.data(), 1, bytes, file);
            std::fflush(file);
        }
    }
    static std::FILE* log = std::fopen("/root/ms/conv_input.log", "w");
    if (log != nullptr && projected.data != nullptr && projected.dtype == DType::BF16) {
        const std::size_t count = std::min<std::size_t>(4, projected.numel());
        std::vector<__nv_bfloat16> head(count);
        if (cudaMemcpy(head.data(), projected.data, count * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
            for (std::size_t i = 0; i < count; ++i) {
                std::fprintf(log, "%.5f ", static_cast<float>(head[i]));
            }
            std::fprintf(log, "\n");
            std::fflush(log);
        }
    }
}

std::size_t gdn_snapshot_workspace_bytes(const Tensor& hidden,
                                         const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    // The GGUF parts carry the same q|k|v geometry the split form does, so both take the shape-keyed
    // snapshot query rather than a QType-keyed one.
    if (std::holds_alternative<GgufGdnInputProjectionPayload>(weights.input_projection)) {
        // The GGUF parts route projects INSIDE this leaf, so its ggml transient is the leaf's to
        // hold as well. Mirrors Variant::gdn_input_projection_snapshot_workspace_capacity_bytes;
        // without the second term the leaf is 625 KB and the projection's 5 MiB stream-k fixup
        // plane dies on it while the CUDA graphs are prepared.
        return std::max(
            kMinimumLeafWorkspaceBytes,
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch, width,
                width) +
                gguf_projection_bytes(16384, batch * width, batch * width));
    }
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t gdn_record_workspace_bytes(const Tensor& hidden,
                                       const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<GgufGdnInputProjectionPayload>(weights.input_projection)) {
        // Same as the snapshot leaf: the ggml transient is spent inside this leaf. Mirrors
        // Variant::gdn_input_projection_record_workspace_capacity_bytes.
        return std::max(
            kMinimumLeafWorkspaceBytes,
            ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch, width,
                width) +
                gguf_projection_bytes(16384, batch * width, batch * width));
    }
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t post_mixer_workspace_bytes(QType gate_up_qtype, QType down_qtype,
                                       ops::LinearPolicy policy, std::int32_t first,
                                       std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gate_up_qtype, 2 * TextConfig::intermediate, TextConfig::hidden, policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down_qtype, TextConfig::hidden, TextConfig::intermediate, policy, first, last));
    }
    return layout.peak_bytes(1);
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile>
Variant::dflash_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window, std::uint32_t) {
    if (capacity == 0 || draft_window == 0 || draft_window > maximum_dflash_draft_tokens) {
        throw std::invalid_argument("invalid DFlash2 graph dimensions");
    }
    // Bounded attention envelopes; each tier owns its topology and can change kernel decomposition.
    auto profiles = graph_profiles_through(capacity - 1, {96, 511, 2047, 8191, 32767});
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        profiles[i].topology_class = static_cast<std::uint32_t>(i);
    }
    return profiles;
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    // GGUF: the parts may differ in format, so there is no fused weight to call the single-parent
    // overload with.
    if (const auto* gguf = std::get_if<GgufAttentionProjectionPayload>(&weights)) {
        ops::attn_input_proj(hidden, gguf->weights, query, gate, key, value, workspace, stream);
        return;
    }
    if (const auto* split = std::get_if<SplitAttentionProjectionPayload>(&weights)) {
        ops::attn_input_proj(hidden, split->query_key, split->gate_value, query, gate, key, value,
                             workspace, stream);
        return;
    }
    const Weight& fused = std::get<FusedAttentionProjectionPayload>(weights).query_key_gate_value;
    ops::attn_input_proj(hidden, fused, query, gate, key, value, text_policy(fused), workspace,
                         stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor packed  = workspace.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, cols});
    ops::linear(hidden, weights.packed, packed, text_policy(weights.packed), workspace, stream);
    Tensor query_heads = query.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor key_heads   = key.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    Tensor gate_heads  = gate.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor value_heads = value.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    ops::mtp_split_attn_in(packed, query_heads, key_heads, gate_heads, value_heads, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    // ops::linear_pair is a W8G32_F16S-only op that takes no workspace, and it validates the weights
    // it is handed. The GGUF MTP block stores key and value as two row ranges of ONE gguf_q6_k
    // object, so on that profile the pair goes through the ordinary workspace-bearing linear twice.
    // The plan reserves the transient for both profiles (see
    // mtp_kv_projection_workspace_capacity_bytes); the pair route simply ignores it.
    if (is_gguf(weights.key.qtype)) {
        ops::linear(hidden, weights.key, key, text_policy(weights.key), workspace, stream);
        ops::linear(hidden, weights.value, value, text_policy(weights.value), workspace, stream);
        return;
    }
    ops::linear_pair(hidden, weights.key, weights.value, key, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear(hidden, weights.query, query, text_policy(weights.query), workspace, stream);
    ops::linear(hidden, weights.output_gate, gate, text_policy(weights.output_gate), workspace, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    // GGUF: q|k|v|z are separate stored objects on most layers (and differ in format), so the
    // projection goes to the parts overload, which projects them all in one ggml call.
    if (const auto* gguf = std::get_if<GgufGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj(hidden, gguf->weights, qkv, output_gate_flat, workspace, stream);
        return;
    }
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj(hidden, split->query_key, split->value_z, qkv, output_gate_flat,
                            workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj(hidden, fused, qkv, output_gate_flat, text_policy(fused), workspace,
                        stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_snapshot_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, split->query_key, split->value_z, conv_weight,
                                          conv_states, valid_columns, initial_slot,
                                          snapshot_base_slot, query, key, value, output_gate_view,
                                          leaf_workspace, stream);
        return;
    }
    if (const auto* gguf = std::get_if<GgufGdnInputProjectionPayload>(&weights.input_projection)) {
        // Compose exactly as the NVFP4 batched path does: project q|k|v into ONE [channels, W, B]
        // plane and z into its own output, then let the projected-conv launch snapshot the conv
        // state from that plane. The parts route writes those two outputs directly, so the only
        // thing this leaf adds is the flatten/view pair the launch's shapes need.
        constexpr std::int32_t kChannels = 2 * TextConfig::key_dim + TextConfig::value_dim;
        const std::int32_t width         = hidden.ne[1];
        const std::int32_t batch         = hidden.ne[2];
        const std::int32_t cols          = width * batch;
        Tensor projected = leaf_workspace.alloc(DType::BF16, {kChannels, cols});
        Tensor gate_flat = output_gate_view.reshape({TextConfig::value_dim, cols});
        ops::gdn_input_proj(hidden.reshape({TextConfig::hidden, cols}), gguf->weights, projected,
                            gate_flat, leaf_workspace, stream);
        ops::detail::gdn_projected_conv_snapshot_launch(
            projected.view({kChannels, width, batch}), conv_weight, conv_states, valid_columns,
            initial_slot, snapshot_base_slot, query, key, value, stream);
        dump_conv_state(conv_states, projected, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_snapshot(hidden, fused, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value,
                                      output_gate_view, text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_record(hidden, split->query_key, split->value_z, conv_weight,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, output_gate_view, leaf_workspace,
                                        stream);
        return;
    }
    if (const auto* gguf = std::get_if<GgufGdnInputProjectionPayload>(&weights.input_projection)) {
        // See gdn_input_projection_snapshot: project into the conv-record plane and z, then let the
        // projected-conv launch record from that plane.
        constexpr std::int32_t kChannels = 2 * TextConfig::key_dim + TextConfig::value_dim;
        const std::int32_t width         = hidden.ne[1];
        const std::int32_t batch         = hidden.ne[2];
        const std::int32_t cols          = width * batch;
        Tensor gate_flat   = output_gate_view.reshape({TextConfig::value_dim, cols});
        Tensor record_flat = conv_record.reshape({kChannels, cols});
        ops::gdn_input_proj(hidden.reshape({TextConfig::hidden, cols}), gguf->weights, record_flat,
                            gate_flat, leaf_workspace, stream);
        ops::detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states,
                                                      valid_columns, initial_slots, query, key, value,
                                                      stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_record(hidden, fused, conv_weight, conv_states, valid_columns,
                                    initial_slots, conv_record, query, key, value, output_gate_view,
                                    text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    ops::linear_add(hidden, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace,
                                          DeviceExecutionView execution) {
    if (const auto* split =
            std::get_if<SplitGdnControlProjectionPayload>(&weights.control_projection)) {
        ops::gdn_norm_gating_proj(residual, norm_weight, eps, split->a_projection,
                                  split->b_projection, weights.a_log, weights.dt_bias, workspace,
                                  hidden, g, beta, execution);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnControlProjectionPayload>(weights.control_projection).a_b_projection;
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, fused, weights.a_log, weights.dt_bias,
                              workspace, hidden, g, beta, execution);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, const ::ninfer::ops::SparseMoeHints&,
                         WorkspaceArena& workspace, cudaStream_t stream, const Tensor* in_norm,
                         float norm_eps) {
    auto scope = workspace.scope();
    // GGUF: gate and up are separate objects with different formats, so the pair goes to the vendored
    // ggml SwiGLU route (which applies silu to the gate half itself) and the norm, when the graph
    // hands the raw residual in, runs as its own op.
    if (const auto* gguf = std::get_if<GgufDenseMlpPayload>(&weights)) {
        const Tensor* use = &hidden;
        Tensor normed;
        if (in_norm != nullptr) {
            normed = workspace.alloc(DType::BF16, {hidden.ne[0], hidden.ne[1]});
            ops::rmsnorm(hidden, *in_norm, norm_eps, true, normed, stream);
            use = &normed;
        }
        Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
        ops::detail::gguf_swiglu(*use, gguf->gate, &gguf->up, activation, workspace, stream);
        ops::linear_add(activation, gguf->down, residual, text_policy(gguf->down), workspace, stream);
        return;
    }
    const auto& fused = std::get<DenseMlpPayload>(weights);
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, fused.gate_up, activation, text_policy(fused.gate_up), workspace,
                       stream, in_norm, norm_eps);
    ops::linear_add(activation, fused.down, residual, text_policy(fused.down), workspace, stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    if (const auto* gguf = std::get_if<GgufDenseMlpPayload>(&weights)) {
        Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
        ops::detail::gguf_swiglu(hidden, gguf->gate, &gguf->up, activation, workspace, stream);
        Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
        ops::linear(activation, gguf->down, delta, text_policy(gguf->down), workspace, stream);
        ops::residual_add(delta, residual, stream);
        return;
    }
    const auto& fused = std::get<DenseMlpPayload>(weights);
    Tensor gate_up    = workspace.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, cols});
    ops::linear(hidden, fused.gate_up, gate_up, text_policy(fused.gate_up), workspace, stream);
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                  gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate), activation,
                  stream);
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    ops::linear(activation, fused.down, delta, text_policy(fused.down), workspace, stream);
    ops::residual_add(delta, residual, stream);
}

// The MTP queries take no weights profile, so they cannot know whether the artifact's MTP block is
// GGUF. Over-reserving is safe -- the arena is sized once for the whole model -- and under-reserving
// is a std::bad_alloc while preparing graphs, which is what `--spec mtp` hit: the q_gate query
// returned 0 unconditionally. Each of these therefore also reserves the GGUF parts route's transient
// bytes, sized for the widest projection in the model rather than the real one.
std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, last});
    return std::max(layout.peak_bytes(1),
                    gguf_projection_bytes(TextConfig::mtp_attention_input_rows, first, last, 1));
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    // Zero was right while this was the W8 pair op, which needs no workspace. It is not right for a
    // GGUF artifact, whose key and value take the ordinary linear route. Like the other MTP queries
    // this one takes no weights profile, so it reserves the GGUF transient unconditionally: the
    // arena is sized once for the whole model and the pair route ignores the extra.
    return gguf_projection_bytes(TextConfig::kv_size, first, last, 1);
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    // Two separate single-part projections, query and output_gate; sized for the model's widest.
    return gguf_projection_bytes(TextConfig::intermediate, first, last, 1);
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        // Fused groupwise-int routes need no transient bytes; the folded ternary port shares this
        // profile and does (see folded_rotation_bytes). This projection consumes the raw hidden
        // state, so the activation is 5120 wide.
        return folded_rotation_bytes(TextConfig::hidden, last);
    case WeightsProfile::Qwen36Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::NVFP4, 14336, TextConfig::hidden, kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, 14336, TextConfig::hidden, kFp8TextPolicy, first, last);
    case WeightsProfile::Qwen38GgufMixed:
        // The GGUF attention projection is a parts list (query|key are row ranges of one object on
        // two layers, and gate/value are always separate objects whose formats differ), so its
        // transient bytes come from the ggml route rather than from a fused-parent query.
        return gguf_projection_bytes(14336, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        // The folded ternary port shares the groupwise-int profile: this projection folds into the
        // residual, and the ternary route needs a [hidden, T] projection scratch plus the
        // [query_size, T] activation rotation buffer. That is a superset of the zero bytes the
        // fused Q5 route needs, so one query serves both artifacts.
        return ops::linear_add_workspace_capacity_bytes(QType::PQ2_0_G128, TextConfig::hidden,
                                                        TextConfig::query_size,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::NVFP4, TextConfig::hidden,
                                                        TextConfig::query_size, kNvfp4TextPolicy,
                                                        first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,
                                                        TextConfig::hidden, TextConfig::query_size,
                                                        kFp8TextPolicy, first, last);
    case WeightsProfile::Qwen38GgufMixed:
        return ops::linear_add_workspace_capacity_bytes(QType::GGUF_IQ2_XXS, TextConfig::hidden,
                                                        TextConfig::query_size,
                                                        ops::LinearPolicy::A16Only, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        // See attention_projection_workspace_capacity_bytes: the folded ternary port shares the
        // groupwise-int profile. This projection also consumes the raw hidden state.
        return folded_rotation_bytes(TextConfig::hidden, last);
    case WeightsProfile::Qwen36Nvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, 16384, TextConfig::hidden,
                                                            kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy, first, last);
    case WeightsProfile::Qwen38GgufMixed:
        // GDN q|k|v|z share one object on 11 of the 48 GDN layers and have z split out on the rest,
        // so this is the parts route too.
        return gguf_projection_bytes(16384, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size,
                            first, last));
    case WeightsProfile::Qwen38Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy,
                            batch_size, first, last));
    case WeightsProfile::Qwen38GgufMixed:
        // The GGUF route has no QType-keyed snapshot profile: it quantizes the activation itself, so
        // the shape-keyed snapshot composition (which is registered for exactly this q/k/v geometry)
        // plus the ggml projection transient is what it needs.
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last) +
                            gguf_projection_bytes(16384, batch_size * first, batch_size * last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size,
                            first, last));
    case WeightsProfile::Qwen38Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy,
                            batch_size, first, last));
    case WeightsProfile::Qwen38GgufMixed:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last) +
                            gguf_projection_bytes(16384, batch_size * first, batch_size * last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        // See attention_output_projection_workspace_capacity_bytes: the ternary route feeds the
        // 6144-wide GDN output into a [hidden, T] scratch and rotates a [value_dim, T] activation.
        return ops::linear_add_workspace_capacity_bytes(QType::PQ2_0_G128, TextConfig::hidden,
                                                        TextConfig::value_dim,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::hidden, TextConfig::value_dim, kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,
                                                        TextConfig::hidden, TextConfig::value_dim,
                                                        kFp8TextPolicy, first, last);
    case WeightsProfile::Qwen38GgufMixed:
        return ops::linear_add_workspace_capacity_bytes(QType::GGUF_IQ2_XXS, TextConfig::hidden,
                                                        TextConfig::value_dim,
                                                        ops::LinearPolicy::A16Only, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    // This query carries no weights profile, and the A/B control projections are among the folded
    // ternary weights, so the folded rotation scratch is reserved unconditionally. It is one
    // activation-sized buffer per token tile, matching what the stage already reserves.
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                              TextConfig::hidden, first, last) +
           folded_rotation_bytes(TextConfig::hidden, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        // The folded ternary port shares the groupwise-int profile: the SwiGLU gate/up parent is
        // projected whole (2 * intermediate rows) and the down projection folds into the residual,
        // so one query reserving both ternary routes serves both artifacts.
        return post_mixer_workspace_bytes(QType::PQ2_0_G128, QType::PQ2_0_G128,
                                          ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
        return post_mixer_workspace_bytes(QType::NVFP4, QType::NVFP4, kNvfp4TextPolicy, first,
                                          last);
    case WeightsProfile::Qwen38Nvfp4: {
        const std::size_t nvfp4 =
            post_mixer_workspace_bytes(QType::NVFP4, QType::NVFP4, kNvfp4TextPolicy, first, last);
        const std::size_t fp8 = post_mixer_workspace_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, QType::FP8_E4M3FN_ROW_BF16S, kFp8TextPolicy, first, last);
        return std::max(nvfp4, fp8);
    }
    case WeightsProfile::Qwen38GgufMixed:
        // gate and up are separate objects with different formats on most layers, so the pair takes
        // the ggml SwiGLU route; post_mixer_workspace_bytes reaches the GGUF branches of both the
        // SwiGLU and the residual linear queries through the qtype.
        return post_mixer_workspace_bytes(QType::GGUF_IQ2_XXS, QType::GGUF_IQ2_XXS,
                                          ops::LinearPolicy::A16Only, first, last);
    }
    throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
}

std::size_t Variant::output_head_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                          std::int32_t first,
                                                          std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        // The folded (rotated-basis) output head maps the activation into the rotated basis from the
        // caller's workspace, and the wide ternary route may also dequantize a weight chunk. Both are
        // in this query; the plan used to reserve only the rotation buffer by hand.
        return ops::linear_workspace_capacity_bytes(QType::PQ2_0_G128, TextConfig::output_rows,
                                                    TextConfig::hidden,
                                                    ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Qwen36Nvfp4:
        return ops::linear_workspace_capacity_bytes(QType::NVFP4, TextConfig::output_rows,
                                                    TextConfig::hidden, kNvfp4TextPolicy, first,
                                                    last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,
                                                    TextConfig::output_rows, TextConfig::hidden,
                                                    kFp8TextPolicy, first, last);
    case WeightsProfile::Qwen38GgufMixed:
        // One part: the artifact stores the head as a single gguf_iq4_xs object, and the ggml route
        // needs an FP32 [248320, T] plane because this leaf hands it a BF16 destination.
        return gguf_projection_bytes(TextConfig::output_rows, first, last, 1);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    // The GGUF branch is a swiglu pair plus a down projection: two parts then one.
    return std::max(layout.peak_bytes(1),
                    gguf_projection_bytes(TextConfig::intermediate, first, last, 2));
}

} // namespace ninfer::targets::qwen3_6_27b::detail
