#pragma once

#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/model_view.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include "ninfer/ops/gguf_projection.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {

inline constexpr std::size_t kTextLayers          = 64;
inline constexpr std::size_t kFullAttentionLayers = 16;
inline constexpr std::size_t kGdnLayers           = 48;

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format          = artifact::NumericFormat::BF16;
    std::uint32_t weight_scale_divisor_bits = 0;
    std::uint32_t input_scale_divisor_bits  = 0;
    // Folded-basis feature permutation for this weight; perm_rep == 1 means "none".
    // Set at bind time from the weight's identity (only /gdn/output carries one).
    // (V100 ternary port.)
    std::int32_t hadamard_perm_hd  = 0;
    std::int32_t hadamard_perm_nk  = 0;
    std::int32_t hadamard_perm_rep = 1;
    // GGUF only: an INT32 [K] gather naming which input element each stored column multiplies.
    // The ModelScope Swift-1.5 artifacts store the GDN output projection's V heads as
    // [repeat, key, 128] while the runtime produces tiled v-heads, so that one matrix needs the
    // permutation; the artifact ships it as an auxiliary object and every GDN layer shares it.
    // `has_input_columns` distinguishes "no gather" from a valid handle index 0.
    bool has_input_columns = false;
    artifact::ObjectHandle input_columns;
};

// Folded (rotated-basis) sign table, present only on artifacts whose weights are folded into a
// rotated basis -- the ternary port. `values` is one contiguous FP32 +-1 block; `widths` names
// consecutive sign rows. `width_offsets` is the prefix sum of `widths`, so a weight can find its
// own block without knowing anything but its input dimension. (V100 ternary port.)
struct HadamardSignsPlan {
    artifact::ObjectHandle values;
    artifact::ObjectHandle widths;
    std::vector<std::pair<std::int32_t, std::uint64_t>> width_offsets;
};

struct MlpPlan {
    WeightPlan gate_up;
    WeightPlan down;
    // GGUF only. The ModelScope Swift-1.5 artifacts store MLP gate and up as SEPARATE objects with
    // different formats (iq1_s/iq1_m, iq2_xxs/iq2_xs, q6_k/q6_k), so they cannot be fused into the
    // 34816-row gate_up parent the other profiles use. When these are set, gate_up is unused and the
    // execution leaf projects the pair with the vendored ggml SwiGLU route.
    std::optional<WeightPlan> gguf_gate;
    std::optional<WeightPlan> gguf_up;
};

struct SplitAttentionProjectionPlan {
    WeightPlan query_key;
    WeightPlan gate_value;
};

struct FusedAttentionProjectionPlan {
    WeightPlan query_key_gate_value;
};

// One GGUF part in the plan: which object, which output tensor (0 = q, 1 = gate, 2 = k, 3 = v), and
// the first row it writes there. Parts may differ in format, which is why this alternative exists.
//
// source_row/source_rows select a ROW SLICE of the bound object: the artifact stores attention
// query and key as two row ranges of one 7168-row object, and one GgufProjectionPart feeds exactly
// one output tensor, so the object is bound once and read as two slices. source_rows == 0 means the
// whole object. A GGUF row is always a whole number of blocks, so a slice is a pointer and row-count
// adjustment and needs no repacking.
struct GgufProjectionPartPlan {
    WeightPlan weight;
    std::int32_t object_rows = 0;
    std::int32_t output      = 0;
    std::int32_t row         = 0;
    std::int32_t source_row  = 0;
    std::int32_t source_rows = 0;
};

struct GgufAttentionProjectionPlan {
    std::vector<GgufProjectionPartPlan> parts;
};

struct GgufGdnInputProjectionPlan {
    std::vector<GgufProjectionPartPlan> parts;
};

struct FullAttentionPlan {
    std::variant<SplitAttentionProjectionPlan, FusedAttentionProjectionPlan,
                 GgufAttentionProjectionPlan>
        projection;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
};

struct SplitGdnInputProjectionPlan {
    WeightPlan query_key;
    WeightPlan value_z;
};

struct FusedGdnInputProjectionPlan {
    WeightPlan query_key_value_z;
};

struct SplitGdnControlProjectionPlan {
    WeightPlan a_projection;
    WeightPlan b_projection;
};

struct FusedGdnControlProjectionPlan {
    WeightPlan a_b_projection;
};

using GdnControlProjectionPlan =
    std::variant<SplitGdnControlProjectionPlan, FusedGdnControlProjectionPlan>;

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    GdnControlProjectionPlan control_projection;
    std::variant<SplitGdnInputProjectionPlan, FusedGdnInputProjectionPlan, GgufGdnInputProjectionPlan>
        input_projection;
    artifact::ObjectHandle norm;
    WeightPlan output;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

struct MtpPlan {
    // The three linear roles carry their format, like every other weight plan in this tree. They used
    // to be bare ObjectHandles, which forced the materializer to name one format for both artifacts;
    // the GGUF profile stores all three as gguf_q6_k and got read as W8G32_F16S.
    WeightPlan input_projection;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
    artifact::ObjectHandle input_norm;
    WeightPlan query_key_gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
    artifact::ObjectHandle final_norm;
};

struct DFlash2DynamicConvPlan {
    artifact::ObjectHandle base_kernel;
    WeightPlan kernel_projection;
};

struct DFlash2LayerPlan {
    artifact::ObjectHandle input_norm;
    DFlash2DynamicConvPlan attention_conv;
    WeightPlan query_key_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan attention_output;
    artifact::ObjectHandle post_attention_norm;
    DFlash2DynamicConvPlan mlp_conv;
    WeightPlan gate_up;
    WeightPlan down;
};

struct DFlash2CandidateSelectorPlan {
    WeightPlan hidden_projection;
    artifact::ObjectHandle predecessor_codebook;
    artifact::ObjectHandle successor_codebook;
};

struct DFlash2Plan {
    WeightPlan feature_projection;
    artifact::ObjectHandle context_norm;
    std::array<DFlash2LayerPlan, qwen3_6::DFlash2Weights::layer_count> layers;
    artifact::ObjectHandle final_norm;
    DFlash2CandidateSelectorPlan candidate_selector;
};

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;

    WeightPlan token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ObjectHandle final_norm;
    WeightPlan output_head;
    artifact::ObjectHandle draft_head;
    artifact::ObjectHandle draft_head_token_ids;
    MtpPlan mtp;
    // Absent unless the artifact stores folded (rotated-basis) weights; the ternary port does.
    std::optional<HadamardSignsPlan> hadamard_signs;
    std::optional<DFlash2Plan> dflash2;

    qwen3_6::VisionBackbonePlan vision_backbone;
    qwen3_6::VisionMergerInputPlan vision_merger_input;
    artifact::ObjectHandle vision_merger_fc2;
    artifact::ObjectHandle vision_merger_fc2_bias;
    qwen3_6::VisionMergerNormPlan vision_merger_norm;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features);

struct DenseMlpPayload {
    Weight gate_up;
    Weight down;
};

// GGUF MLP: gate and up are separate stored objects with different formats, so the pair goes to the
// vendored ggml SwiGLU route rather than to the fused gate_up path.
struct GgufDenseMlpPayload {
    Weight gate;
    Weight up;
    Weight down;
};

using DensePostMixerPayload = std::variant<DenseMlpPayload, GgufDenseMlpPayload>;

struct SplitAttentionProjectionPayload {
    Weight query_key;
    Weight gate_value;
};

struct FusedAttentionProjectionPayload {
    Weight query_key_gate_value;
};

struct GgufAttentionProjectionPayload {
    ops::GgufProjectionWeights weights;
};

using FullAttentionProjectionPayload =
    std::variant<SplitAttentionProjectionPayload, FusedAttentionProjectionPayload,
                 GgufAttentionProjectionPayload>;

struct SplitGdnInputProjectionPayload {
    Weight query_key;
    Weight value_z;
};

struct FusedGdnInputProjectionPayload {
    Weight query_key_value_z;
};

struct GgufGdnInputProjectionPayload {
    ops::GgufProjectionWeights weights;
};

using GdnInputProjectionPayload =
    std::variant<SplitGdnInputProjectionPayload, FusedGdnInputProjectionPayload,
                 GgufGdnInputProjectionPayload>;

struct SplitGdnControlProjectionPayload {
    Weight a_projection;
    Weight b_projection;
};

struct FusedGdnControlProjectionPayload {
    Weight a_b_projection;
};

using GdnControlProjectionPayload =
    std::variant<SplitGdnControlProjectionPayload, FusedGdnControlProjectionPayload>;

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    GdnControlProjectionPayload control_projection;
    GdnInputProjectionPayload input_projection;
};

struct MtpAttentionPayload {
    Weight packed;
    Weight query;
    Weight key;
    Weight output_gate;
    Weight value;
};

using RuntimeModelView =
    qwen3_6::ModelView<FullAttentionProjectionPayload, GdnProjectionPayload, DensePostMixerPayload,
                       MtpAttentionPayload, DensePostMixerPayload, qwen3_6::DFlash2Weights,
                       kFullAttentionLayers, kGdnLayers>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    qwen3_6::FrontendResources frontend;
    RuntimeModelView runtime;
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan plan,
         artifact::MaterializedArtifact materialized)
        : weights_profile(weights_profile_in), data(std::move(plan), std::move(materialized)) {}

    WeightsProfile weights_profile;
    LoadedModelData data;
};

} // namespace ninfer::targets::qwen3_6_27b::detail
