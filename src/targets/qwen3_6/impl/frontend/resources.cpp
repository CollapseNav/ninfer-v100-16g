#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <limits>
#include <string>

namespace ninfer::targets::qwen3_6 {
namespace {

// The two pixel-processor configs are vision-only resources: a package that keeps its vision tower
// outside the container (the Coder390 GGUF-NInfer exports ship vision-bf16.safetensors beside the
// .ninfer) does not carry them, and upstream reads them only when vision is enabled. Binding them
// unconditionally would make such an artifact unloadable.
constexpr artifact::ObjectHandle kAbsentResource{std::numeric_limits<std::size_t>::max()};

artifact::ObjectHandle bind_optional_raw_resource(artifact::Binder& binder, std::string_view name) {
    if (!binder.contains(name)) { return kAbsentResource; }
    return artifact::bind_raw_resource(binder, name);
}

std::string take_string(artifact::MaterializedArtifact& materialized,
                        artifact::ObjectHandle handle) {
    if (handle.index == kAbsentResource.index) { return {}; }
    const auto bytes = materialized.take_resource_bytes(handle);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

} // namespace

FrontendResourcePlan bind_frontend_resources(artifact::Binder& binder) {
    return FrontendResourcePlan{
        .tokenizer_json = artifact::bind_raw_resource(binder, "frontend/tokenizer.json"),
        .tokenizer_config_json =
            artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json"),
        .chat_template_jinja = artifact::bind_raw_resource(binder, "frontend/chat_template.jinja"),
        .generation_config_json =
            artifact::bind_raw_resource(binder, "frontend/generation_config.json"),
        .preprocessor_config_json =
            bind_optional_raw_resource(binder, "frontend/preprocessor_config.json"),
        .video_preprocessor_config_json =
            bind_optional_raw_resource(binder, "frontend/video_preprocessor_config.json"),
    };
}

FrontendResources take_frontend_resources(artifact::MaterializedArtifact& materialized,
                                          const FrontendResourcePlan& plan) {
    return FrontendResources{
        .tokenizer_json           = take_string(materialized, plan.tokenizer_json),
        .tokenizer_config_json    = take_string(materialized, plan.tokenizer_config_json),
        .chat_template_jinja      = take_string(materialized, plan.chat_template_jinja),
        .generation_config_json   = take_string(materialized, plan.generation_config_json),
        .preprocessor_config_json = take_string(materialized, plan.preprocessor_config_json),
        .video_preprocessor_config_json =
            take_string(materialized, plan.video_preprocessor_config_json),
    };
}

} // namespace ninfer::targets::qwen3_6
