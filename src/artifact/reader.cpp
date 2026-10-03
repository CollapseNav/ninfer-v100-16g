#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

constexpr std::array<std::byte, 8> kMagicV2 = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};
constexpr std::array<std::byte, 8> kMagicV3 = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{3},
};
constexpr std::uint64_t kPayloadAlignment = 4096;

// v2 header = 16 bytes (magic + json_length); v3 adds a 16-byte UUID = 32 total.
inline std::uint64_t header_bytes_for(bool is_v3) { return is_v3 ? 32ULL : 16ULL; }

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(data[i])) << (i * 8);
    }
    return value;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

NumericFormat parse_format(std::string_view name) {
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    if (name == "NVFP4") { return NumericFormat::NVFP4; }
    if (name == "FP8_E4M3FN_ROW_BF16S") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    // This fork's ternary formats (absent upstream) and the GGUF block families the ModelScope
    // Swift-1.5 artifacts store. A GGUF format parses so the container layer can describe it, but no
    // kernel consumes it yet; see docs/ternary-port.md.
    if (name == "PTQ1_0_G128") { return NumericFormat::PTQ1_0_G128; }
    if (name == "PQ2_0_G128") { return NumericFormat::PQ2_0_G128; }
    if (name == "GGUF_Q2_K") { return NumericFormat::GGUF_Q2_K; }
    if (name == "GGUF_Q4_K") { return NumericFormat::GGUF_Q4_K; }
    if (name == "GGUF_Q6_K") { return NumericFormat::GGUF_Q6_K; }
    if (name == "GGUF_IQ1_S") { return NumericFormat::GGUF_IQ1_S; }
    if (name == "GGUF_IQ1_M") { return NumericFormat::GGUF_IQ1_M; }
    if (name == "GGUF_IQ2_XXS") { return NumericFormat::GGUF_IQ2_XXS; }
    if (name == "GGUF_IQ2_XS") { return NumericFormat::GGUF_IQ2_XS; }
    if (name == "GGUF_IQ2_S") { return NumericFormat::GGUF_IQ2_S; }
    if (name == "GGUF_IQ3_XXS") { return NumericFormat::GGUF_IQ3_XXS; }
    if (name == "GGUF_IQ3_S") { return NumericFormat::GGUF_IQ3_S; }
    if (name == "GGUF_IQ4_XS") { return NumericFormat::GGUF_IQ4_XS; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_layout(std::string_view name) {
    if (name == "contiguous-le-v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row-split-k128-v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "blockscale-k16-m128x4-v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    if (name == "row-scale-v1") { return StorageLayout::RowScaleV1; }
    // GGUF writes its blocks contiguously and the artifact declares that layout directly, so the
    // v3 and fork spellings are the same string.
    if (name == "gguf_blocks_v1") { return StorageLayout::GgufBlocksV1; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding(std::string_view name) {
    if (name == "raw-bytes-v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

// Upstream v3 writes a lowercase snake_case vocabulary for format/layout/encoding
// strings; the fork runtime keeps its v2 vocabulary.  Map on read so that
// parse_format/parse_layout/parse_encoding see fork-native strings.  Values
// already in the fork vocabulary (fork-side v2->v3 conversions) pass through.
std::string_view map_v3_format(std::string_view value) {
    if (value == "bf16") { return "BF16"; }
    if (value == "fp32") { return "FP32"; }
    if (value == "int32") { return "I32"; }
    if (value == "nvfp4") { return "NVFP4"; }
    if (value == "q4_g64_fp16") { return "Q4G64_F16S"; }
    if (value == "q5_g64_fp16") { return "Q5G64_F16S"; }
    if (value == "q6_g64_fp16") { return "Q6G64_F16S"; }
    if (value == "q8_g32_fp16") { return "W8G32_F16S"; }
    if (value == "fp8_e4m3fn_row_bf16") { return "FP8_E4M3FN_ROW_BF16S"; }
    if (value == "gguf_q2_k") { return "GGUF_Q2_K"; }
    if (value == "gguf_q4_k") { return "GGUF_Q4_K"; }
    if (value == "gguf_q6_k") { return "GGUF_Q6_K"; }
    if (value == "gguf_iq1_s") { return "GGUF_IQ1_S"; }
    if (value == "gguf_iq1_m") { return "GGUF_IQ1_M"; }
    if (value == "gguf_iq2_xxs") { return "GGUF_IQ2_XXS"; }
    if (value == "gguf_iq2_xs") { return "GGUF_IQ2_XS"; }
    if (value == "gguf_iq2_s") { return "GGUF_IQ2_S"; }
    if (value == "gguf_iq3_xxs") { return "GGUF_IQ3_XXS"; }
    if (value == "gguf_iq3_s") { return "GGUF_IQ3_S"; }
    if (value == "gguf_iq4_xs") { return "GGUF_IQ4_XS"; }
    // "t2_g128_fp16" is deliberately NOT mapped onto PQ2_0_G128. The container geometry is
    // identical (32 code bytes + 2 scale bytes per 128 weights) but the 2-bit dictionaries differ:
    // T2 is two's-complement signed (00->0, 01->+1, 11->-1) while PQ2_0 is code-1 over {0,1,2}.
    // Mapping it would decode every ternary weight with the wrong dictionary and produce a model
    // that loads and generates plausible-looking garbage. Leaving it unmapped makes the reader
    // reject the artifact with "unknown tensor format: t2_g128_fp16" instead.
    return value;
}

std::string_view map_v3_layout(std::string_view value) {
    if (value == "contiguous_le_v1") { return "contiguous-le-v1"; }
    if (value == "row_split_k128_v1") { return "row-split-k128-v1"; }
    if (value == "row_scale_v1") { return "row-scale-v1"; }
    if (value == "block_scale_k16_m128x4_v1") { return "blockscale-k16-m128x4-v1"; }
    return value;
}

std::string_view map_v3_encoding(std::string_view value) {
    if (value == "raw_bytes_v1") { return "raw-bytes-v1"; }
    return value;
}

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry");

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {name, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
#ifdef _WIN32
        file_ = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "CreateFileW " + path.string());
        }
        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(file_, &file_size)) {
            const auto error = ::GetLastError();
            ::CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "GetFileSizeEx " + path.string());
        }
        if (file_size.QuadPart < 0 ||
            static_cast<std::uint64_t>(file_size.QuadPart) >
                std::numeric_limits<std::size_t>::max()) {
            ::CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            throw ArtifactError("artifact size does not fit the process address space");
        }
        size_ = static_cast<std::size_t>(file_size.QuadPart);
        if (size_ != 0) {
            mapping_ = ::CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping_ == nullptr) {
                const auto error = ::GetLastError();
                ::CloseHandle(file_);
                file_ = INVALID_HANDLE_VALUE;
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "CreateFileMappingW " + path.string());
            }
            data_ = static_cast<const std::byte*>(
                ::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
            if (data_ == nullptr) {
                const auto error = ::GetLastError();
                ::CloseHandle(mapping_);
                ::CloseHandle(file_);
                mapping_ = nullptr;
                file_    = INVALID_HANDLE_VALUE;
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "MapViewOfFile " + path.string());
            }
        }
#else
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
#endif
    }

    ~MappedFile() {
#ifdef _WIN32
        if (data_ != nullptr) { ::UnmapViewOfFile(data_); }
        if (mapping_ != nullptr) { ::CloseHandle(mapping_); }
        if (file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(file_); }
#else
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
#endif
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
#ifdef _WIN32
        if (destination.size() > std::numeric_limits<DWORD>::max()) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }
        OVERLAPPED operation{};
        operation.Offset     = static_cast<DWORD>(absolute_offset);
        operation.OffsetHigh = static_cast<DWORD>(absolute_offset >> 32U);
        operation.hEvent     = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (operation.hEvent == nullptr) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "create direct artifact read event");
        }
        DWORD bytes           = 0;
        BOOL complete = ::ReadFile(file_, destination.data(), static_cast<DWORD>(destination.size()),
                                   &bytes, &operation);
        DWORD error   = complete ? ERROR_SUCCESS : ::GetLastError();
        if (!complete && error == ERROR_IO_PENDING) {
            complete = ::GetOverlappedResult(file_, &operation, &bytes, TRUE);
            error    = complete ? ERROR_SUCCESS : ::GetLastError();
        }
        ::CloseHandle(operation.hEvent);
        if (!complete && error != ERROR_HANDLE_EOF) {
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "direct artifact read");
        }
        return bytes;
#else
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
#endif
    }

private:
#ifdef _WIN32
    HANDLE file_    = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_                = -1;
#endif
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

} // namespace

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

// Upstream v3 exposes fused tensors (gate|up, query|key|value|gate, ...) as one
// physical object sliced by "parts" bindings whose logical names are the split
// component names.  The fork binder instead binds the fused tensor under the
// fused base name.  Each family lists the split suffixes; a family resolves for
// a directory prefix when every component exists there as a single-part binding
// over one and the same physical object.
struct FusedFamily {
    std::string_view base;
    std::array<std::string_view, 4> components;
};
constexpr std::array<FusedFamily, 7> kFusedFamilies = {{
    {"gate_up", {"gate", "up", "", ""}},
    {"query_key_gate_value", {"query", "key", "value", "gate"}},
    {"query_key_value_z", {"query", "key", "value", "z"}},
    {"a_b_projection", {"a_projection", "b_projection", "", ""}},
    {"qkv", {"query", "key", "value", ""}},
    {"query_key_value", {"query", "key", "value", ""}},
    {"qkv_bias", {"query_bias", "key_bias", "value_bias", ""}},
}};

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path) : file(path) {
        if (file.size() < 16) {
            throw ArtifactError("artifact is shorter than the minimum header");
        }
        const bool is_v3 = std::equal(kMagicV3.begin(), kMagicV3.end(), file.data());
        const bool is_v2 = !is_v3 && std::equal(kMagicV2.begin(), kMagicV2.end(), file.data());
        if (!is_v2 && !is_v3) {
            throw ArtifactError("artifact magic is not NInfer v2 or v3");
        }

        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto header = header_bytes_for(is_v3);
        const auto metadata_end = checked_add(header, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + header);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        // Local storage for v3 (which needs id->name normalization per object).
        std::vector<Json> v3_objects_storage;

        // Pointer to the array we'll iterate. For v2 this points into directory;
        // for v3 we build v3_objects_storage with id copied into name.
        const Json* raw_objects = nullptr;

        if (is_v2) {
            static constexpr std::array root_members = {"identity", "objects"};
            require_members(directory, root_members, "directory root");
            const auto& raw_identity                     = directory.at("identity");
            static constexpr std::array identity_members = {"model_id", "weights_id"};
            require_members(raw_identity, identity_members, "artifact identity");
            identity.model_id   = require_string(raw_identity.at("model_id"), "model_id");
            identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");
            raw_objects = &directory.at("objects");
        } else {
            // v3 root schema: {components, objects, bindings, uses, metadata, provenance, files}
            if (!directory.contains("objects")) {
                throw ArtifactError("v3 directory is missing objects array");
            }
            // Recover identity for logging; metadata.name is the model_id.
            if (directory.contains("metadata") && directory.at("metadata").is_object()) {
                const auto& meta = directory.at("metadata");
                if (meta.contains("model_id") && meta.at("model_id").is_string()) {
                    identity.model_id = require_string(meta.at("model_id"), "metadata.model_id");
                } else if (meta.contains("name") && meta.at("name").is_string()) {
                    identity.model_id = require_string(meta.at("name"), "metadata.name");
                }
            }
            // v3 artifacts name themselves after the release rather than after the checkpoint, e.g.
            // "swift-1.5-qwen3.8-27b-gsq-rco-iq2xs" for what is a qwen3.8-27b. The compiled target
            // registry keys on the checkpoint-native id, so normalize by the ids this fork
            // registers. Longest id first so a prefix cannot shadow a longer one.
            if (!identity.model_id.empty()) {
                static constexpr std::array<std::string_view, 3> kRegisteredModelIds = {
                    "qwen3.6-35b-a3b", "qwen3.8-27b", "qwen3.6-27b"};
                for (const auto registered : kRegisteredModelIds) {
                    if (identity.model_id.find(registered) != std::string::npos) {
                        identity.model_id = std::string(registered);
                        break;
                    }
                }
            }
            if (directory.contains("provenance") && directory.at("provenance").is_object()
                && directory.at("provenance").contains("upgraded_from")
                && directory.at("provenance").at("upgraded_from").is_object()) {
                const auto& up = directory.at("provenance").at("upgraded_from");
                if (up.contains("weights_id") && up.at("weights_id").is_string()) {
                    identity.weights_id = require_string(up.at("weights_id"), "upgraded_from.weights_id");
                }
            }
            if (identity.weights_id.empty()) {
                // Native upstream v3 artifacts carry no weights_id; recover it
                // from the converter recipe (e.g. "qwen3_8_27b_nvfp4"), then
                // from the tensor formats themselves.
                if (directory.contains("provenance") && directory.at("provenance").is_object()
                    && directory.at("provenance").contains("recipe")
                    && directory.at("provenance").at("recipe").is_string()) {
                    const auto& recipe =
                        directory.at("provenance").at("recipe").get_ref<const std::string&>();
                    if (recipe.find("groupwise") != std::string::npos) {
                        identity.weights_id = "groupwise-int";
                    } else if (recipe.find("nvfp4") != std::string::npos) {
                        identity.weights_id = "nvfp4";
                    } else if (recipe.find("gguf") != std::string::npos) {
                        // The GGUF block families. One weights_id covers the whole published family
                        // (iq2_s/iq2_xs/iq3_s/iq3_xxs) because the parameter inventory, names and
                        // fusion are identical across them and only the per-role format assignment
                        // differs; the binder accepts the declared GGUF format rather than a fixed
                        // one, the same loosening the ternary port already uses for row-split
                        // formats.
                        identity.weights_id = "gguf-mixed";
                    }
                }
                if (identity.weights_id.empty()) {
                    const auto& objects_array = directory.at("objects");
                    for (const auto& obj : objects_array) {
                        if (!obj.is_object()) { continue; }
                        const auto kind = obj.find("kind");
                        const auto fmt  = obj.find("format");
                        if (kind == obj.end() || !kind->is_string()
                            || kind->get_ref<const std::string&>() != "tensor"
                            || fmt == obj.end() || !fmt->is_string()) {
                            continue;
                        }
                        const auto& format = fmt->get_ref<const std::string&>();
                        if (format == "nvfp4" || format == "NVFP4") {
                            identity.weights_id = "nvfp4";
                            break;
                        }
                        if (format.rfind("gguf_", 0) == 0 || format.rfind("GGUF_", 0) == 0) {
                            identity.weights_id = "gguf-mixed";
                            break;
                        }
                    }
                }
            }
            // v3 objects use "id"; parse_object/parse_tensor expect "name".
            const auto& dir_objects = directory.at("objects");
            if (!dir_objects.is_array() || dir_objects.empty()) {
                throw ArtifactError("v3 objects must be a nonempty array");
            }
            v3_objects_storage.reserve(dir_objects.size());
            for (const auto& obj : dir_objects) {
                Json copy = obj;
                if (copy.contains("id") && !copy.contains("name")) {
                    copy["name"] = copy["id"];
                }
                // parse_tensor/parse_resource validate the exact v2 member set;
                // drop the v3-only "id" after copying it into "name".
                copy.erase("id");
                for (const auto* key : {"format", "layout", "encoding"}) {
                    const auto field = copy.find(key);
                    if (field == copy.end() || !field->is_string()) { continue; }
                    const auto& raw   = field->get_ref<const std::string&>();
                    const auto mapped = std::string_view(key) == "format"
                                            ? map_v3_format(raw)
                                        : std::string_view(key) == "layout"
                                            ? map_v3_layout(raw)
                                            : map_v3_encoding(raw);
                    if (mapped != raw) { copy[key] = std::string(mapped); }
                }
                v3_objects_storage.push_back(std::move(copy));
            }
            raw_objects = nullptr;  // will iterate v3_objects_storage instead
        }

        // Iteration facade: v2 iterates the JSON array directly; v3 iterates the normalized copy.
        // We merge both into a single code path via a small helper.
        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;

        auto process_object = [&](const Json& raw_object) {
            auto object          = parse_object(raw_object);
            const auto name      = object_name(object);
            const auto offset    = object_offset(object);
            const auto bytes     = object_bytes(object);
            const auto alignment = std::visit(
                [](const auto& descriptor) {
                    using Descriptor = std::decay_t<decltype(descriptor)>;
                    if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                        return tensor_alignment(descriptor.layout);
                    } else {
                        return resource_alignment(descriptor.encoding);
                    }
                },
                object);

            if (offset < cursor) {
                throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
            }
            if (offset % alignment != 0) {
                throw ArtifactError("object " + std::string(name) + " is not " +
                                    std::to_string(alignment) + "-byte aligned");
            }
            const auto end = checked_add(offset, bytes, "object payload range");
            if (end > payload_bytes) {
                throw ArtifactError("object " + std::string(name) + " extends beyond the file");
            }
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(std::string(name), object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
            entries.push_back(std::move(object));
            cursor = end;
        };

        if (is_v2) {
            for (const auto& raw_object : *raw_objects) { process_object(raw_object); }
        } else {
            for (const auto& raw_object : v3_objects_storage) { process_object(raw_object); }
            build_v3_logical_layer(directory);
        }
    }

    // Builds the fork-namespace view of an upstream v3 directory: logical
    // aliases mapping every name the fork binder may request (identity simple
    // bindings, fused family names, activation input divisors, frontend
    // resources, and the fixed namespace renames) onto physical entries.
    // See kFusedFamilies for the fused family table.
    void build_v3_logical_layer(const Json& directory) {
        const auto physical = [this](std::string_view object_id) -> const std::size_t* {
            const auto it = index.find(object_id);
            return it == index.end() ? nullptr : &it->second;
        };
        auto add_logical = [&](const std::string& fork_name, std::size_t entry_index,
                               bool first_wins) {
            const auto [it, inserted] = logical_index.emplace(fork_name, entry_index);
            if (!inserted && it->second != entry_index && !first_wins) {
                throw ArtifactError("conflicting v3 logical alias: " + fork_name);
            }
        };

        // Applies the fixed fork-namespace renames to a logical name; returns an
        // empty string when no rule applies.
        auto fork_rename = [](std::string_view name) -> std::string {
            if (name == "proposal/head") { return "text/draft_head"; }
            if (name == "proposal/token_ids") { return "text/draft_head_token_ids"; }
            if (name.rfind("mtp/layers/0/", 0) == 0) {
                return "mtp/layer/" + std::string(name.substr(std::strlen("mtp/layers/0/")));
            }
            if (name.rfind("vision/layers/", 0) == 0) {
                static constexpr std::array<std::pair<std::string_view, std::string_view>, 4>
                    norm_rules = {{
                        {"norm1_weight", "norm1/weight"},
                        {"norm1_bias", "norm1/bias"},
                        {"norm2_weight", "norm2/weight"},
                        {"norm2_bias", "norm2/bias"},
                    }};
                for (const auto& [from, to] : norm_rules) {
                    std::string needle = "/";
                    needle += from;
                    if (name.size() > needle.size()
                        && name.compare(name.size() - needle.size(), needle.size(), needle) == 0) {
                        std::string out(name.substr(0, name.size() - needle.size()));
                        out += "/";
                        out += to;
                        return out;
                    }
                }
                return {};
            }
            if (name == "vision/merger/norm_weight") { return "vision/merger/norm/weight"; }
            if (name == "vision/merger/norm_bias") { return "vision/merger/norm/bias"; }
            return {};
        };

        struct BindingEntry {
            bool simple       = true;
            std::string object;  // simple: the bound object; parts: first part object
            std::size_t part_count = 0;
            // The first part's element range, when the binding names one. `range` is an element
            // offset, so dividing by the object's column count gives rows.
            std::uint64_t range_begin = 0;
            std::uint64_t range_end   = 0;
        };
        std::unordered_map<std::string, BindingEntry, TransparentStringHash, std::equal_to<>>
            bindings;
        if (directory.contains("bindings") && directory.at("bindings").is_object()) {
            for (const auto& [name, value] : directory.at("bindings").items()) {
                if (!value.is_object()) {
                    throw ArtifactError("v3 binding entry must be an object: " + name);
                }
                BindingEntry parsed;
                if (value.contains("object")) {
                    if (!value.at("object").is_string()) {
                        throw ArtifactError("v3 binding object must be a string: " + name);
                    }
                    parsed.simple     = true;
                    parsed.object     = value.at("object").get<std::string>();
                    parsed.part_count = 1;
                } else if (value.contains("parts")) {
                    const auto& parts = value.at("parts");
                    if (!parts.is_array() || parts.empty()) {
                        throw ArtifactError("v3 binding parts must be a nonempty array: " + name);
                    }
                    parsed.simple     = false;
                    parsed.part_count = parts.size();
                    if (!parts.at(0).is_object() || !parts.at(0).contains("object")
                        || !parts.at(0).at("object").is_string()) {
                        throw ArtifactError("v3 binding part must reference an object: " + name);
                    }
                    parsed.object = parts.at(0).at("object").get<std::string>();
                    if (parts.at(0).contains("range") && parts.at(0).at("range").is_array() &&
                        parts.at(0).at("range").size() == 2 &&
                        parts.at(0).at("range").at(0).is_number_unsigned() &&
                        parts.at(0).at("range").at(1).is_number_unsigned()) {
                        parsed.range_begin = parts.at(0).at("range").at(0).get<std::uint64_t>();
                        parsed.range_end   = parts.at(0).at("range").at(1).get<std::uint64_t>();
                    }
                } else {
                    throw ArtifactError("v3 binding entry has neither object nor parts: " + name);
                }
                bindings.emplace(name, std::move(parsed));
            }
        }

        // Returns the shared object id of a fused family under a directory
        // prefix, or an empty string when the family does not fully resolve.
        auto resolve_family = [&](const std::string& prefix,
                                  const FusedFamily& family) -> std::string {
            std::string object;
            for (const auto component : family.components) {
                if (component.empty()) { break; }
                const auto it = bindings.find(prefix + std::string(component));
                if (it == bindings.end() || it->second.simple || it->second.part_count != 1) {
                    return {};
                }
                if (object.empty()) {
                    object = it->second.object;
                } else if (object != it->second.object) {
                    return {};
                }
            }
            return object;
        };

        // Simple bindings: identity plus the fixed fork-namespace renames.
        for (const auto& [name, entry] : bindings) {
            if (!entry.simple) { continue; }
            const auto* entry_index = physical(entry.object);
            if (entry_index == nullptr) {
                throw ArtifactError("v3 binding references unknown object: " + entry.object);
            }
            add_logical(name, *entry_index, false);
            if (std::string forked = fork_rename(name); !forked.empty()) {
                add_logical(forked, *entry_index, false);
            }
        }

        // Fused families: split component bindings expose one physical object;
        // the fork binds that object under the fused base name.
        for (const auto& [name, entry] : bindings) {
            if (entry.simple) { continue; }
            const auto slash = name.rfind('/');
            if (slash == std::string::npos) { continue; }
            const std::string prefix = name.substr(0, slash + 1);
            const std::string suffix = name.substr(slash + 1);
            for (const auto& family : kFusedFamilies) {
                const bool member =
                    std::find(family.components.begin(), family.components.end(), suffix)
                    != family.components.end();
                if (!member) { continue; }
                const std::string object = resolve_family(prefix, family);
                if (object.empty()) { continue; }
                const auto* entry_index = physical(object);
                if (entry_index == nullptr) {
                    throw ArtifactError("v3 fused family references unknown object: " + object);
                }
                const std::string fused_name = prefix + std::string(family.base);
                add_logical(fused_name, *entry_index, false);
                if (std::string renamed = fork_rename(fused_name); !renamed.empty()) {
                    add_logical(renamed, *entry_index, false);
                }
            }
        }

        // Activation input divisors: uses[].auxiliaries.activation_input_divisor
        // names an FP32 scalar auxiliary; the fork expects it under
        // "{fused_owner}_projection/input_scale_divisor".
        if (directory.contains("uses") && directory.at("uses").is_array()) {
            for (const auto& use : directory.at("uses")) {
                if (!use.is_object()) { continue; }
                const auto param_it = use.find("parameter");
                if (param_it == use.end() || !param_it->is_string()) { continue; }
                const auto aux_it = use.find("auxiliaries");
                if (aux_it == use.end() || !aux_it->is_object()) { continue; }
                const auto div_it = aux_it->find("activation_input_divisor");
                if (div_it == aux_it->end() || !div_it->is_object()) { continue; }
                const auto obj_it = div_it->find("object");
                if (obj_it == div_it->end() || !obj_it->is_string()) { continue; }
                const auto& parameter = param_it->get_ref<const std::string&>();
                const auto& auxiliary = obj_it->get_ref<const std::string&>();
                const auto* entry_index = physical(auxiliary);
                if (entry_index == nullptr) {
                    throw ArtifactError("v3 use references unknown auxiliary: " + auxiliary);
                }
                std::string fork_divisor;
                const auto slash = parameter.rfind('/');
                if (slash != std::string::npos) {
                    const std::string prefix = parameter.substr(0, slash + 1);
                    const std::string suffix = parameter.substr(slash + 1);
                    for (const auto& family : kFusedFamilies) {
                        const bool member =
                            std::find(family.components.begin(), family.components.end(), suffix)
                            != family.components.end();
                        if (!member) { continue; }
                        if (resolve_family(prefix, family).empty()) { continue; }
                        fork_divisor = prefix + std::string(family.base)
                                       + "_projection/input_scale_divisor";
                        break;
                    }
                }
                if (fork_divisor.empty()) {
                    fork_divisor = parameter + "_projection/input_scale_divisor";
                }
                // gate|up parts may each carry an equal-valued divisor; keep the
                // first mapping for a fused owner instead of throwing.
                add_logical(fork_divisor, *entry_index, true);
                if (std::string renamed = fork_rename(fork_divisor); !renamed.empty()) {
                    add_logical(renamed, *entry_index, true);
                }
            }
        }

        // Frontend resources: components.{text,vision}.resources -> frontend/.
        if (directory.contains("components") && directory.at("components").is_object()) {
            const auto& components = directory.at("components");
            for (const char* component_name : {"text", "vision"}) {
                const auto comp_it = components.find(component_name);
                if (comp_it == components.end() || !comp_it->is_object()) { continue; }
                const auto res_it = comp_it->find("resources");
                if (res_it == comp_it->end() || !res_it->is_object()) { continue; }
                for (const auto& [short_name, reference] : res_it->items()) {
                    if (!reference.is_string()) { continue; }
                    const auto* entry_index = physical(reference.get_ref<const std::string&>());
                    if (entry_index == nullptr) {
                        throw ArtifactError("v3 component references unknown resource: "
                                            + reference.get<std::string>());
                    }
                    add_logical(std::string("frontend/") + short_name, *entry_index, false);
                }
            }
        }

        // Expose only the objects reachable through the fork-namespace view:
        // upstream v3 may carry physical objects the fork never binds (e.g.
        // the redundant halves of duplicated activation divisors), and
        // Binder::finish() enforces that every exposed object is consumed.
        // Artifacts without bindings (fork-side v2->v3 conversions) name their
        // objects physically and are left untouched.
        if (!bindings.empty()) {
            constexpr std::size_t kUnmapped = std::numeric_limits<std::size_t>::max();
            std::vector<bool> reachable(entries.size(), false);
            for (const auto& [name, old_index] : logical_index) { reachable[old_index] = true; }
            // An object a v3 binding addresses by ROW RANGE is reachable as well: a model binder
            // reaches it through Reader::find_slice rather than through a logical alias, and the
            // fused-family rule does not cover every such binding (attention query|key share an
            // object that no family name resolves, for instance).
            for (const auto& [name, entry] : bindings) {
                if (entry.simple || entry.part_count != 1) { continue; }
                const auto found = index.find(entry.object);
                if (found != index.end()) { reachable[found->second] = true; }
            }
            // And so is any object a `uses` entry names as an auxiliary. Those carry per-weight
            // tables rather than weights -- the ModelScope Swift-1.5 artifacts ship the GDN output
            // projection's input_columns permutation as auxiliary/000000, a single INT32 [K] object
            // shared by every GDN layer -- and a model binder binds them by object id.
            if (directory.contains("uses") && directory.at("uses").is_array()) {
                for (const auto& use : directory.at("uses")) {
                    const auto auxiliaries = use.find("auxiliaries");
                    if (auxiliaries == use.end() || !auxiliaries->is_object()) { continue; }
                    for (const auto& [role, reference] : auxiliaries->items()) {
                        if (!reference.is_object()) { continue; }
                        const auto object_id = reference.find("object");
                        if (object_id == reference.end() || !object_id->is_string()) { continue; }
                        const auto found =
                            index.find(object_id->get_ref<const std::string&>());
                        if (found != index.end()) { reachable[found->second] = true; }
                    }
                }
            }
            std::vector<std::size_t> remap(entries.size(), kUnmapped);
            std::vector<ObjectDescriptor> kept;
            kept.reserve(logical_index.size());
            for (std::size_t old = 0; old < entries.size(); ++old) {
                if (!reachable[old]) { continue; }
                remap[old] = kept.size();
                kept.push_back(std::move(entries[old]));
            }
            std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>>
                physical;
            physical.reserve(kept.size());
            for (std::size_t i = 0; i < kept.size(); ++i) {
                physical.emplace(std::string(object_name(kept[i])), i);
            }
            std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>>
                logical;
            logical.reserve(logical_index.size());
            for (const auto& [name, old_index] : logical_index) {
                logical.emplace(name, remap[old_index]);
            }
            entries       = std::move(kept);
            index         = std::move(physical);
            logical_index = std::move(logical);
        }

        // Row ranges, recorded after the remap so the entry indices are final and keyed by the
        // binding's own v3 name because that is what a model binder asks for. Only single-part
        // bindings are representable: a multi-part binding is a concatenation, not a slice.
        for (const auto& [name, entry] : bindings) {
            if (entry.simple || entry.part_count != 1 || entry.range_end <= entry.range_begin) {
                continue;
            }
            const auto found = index.find(entry.object);
            if (found == index.end()) { continue; }
            const auto& object  = entries[found->second];
            const auto* tensor  = std::get_if<TensorDescriptor>(&object);
            if (tensor == nullptr || tensor->shape.size() != 2 || tensor->shape[1] == 0) {
                continue;
            }
            const std::uint64_t columns = tensor->shape[1];
            ObjectSlice slice;
            slice.object    = &object;
            slice.first_row = entry.range_begin / columns;
            slice.rows      = (entry.range_end - entry.range_begin) / columns;
            if (slice.rows != 0) { slice_index.emplace(name, slice); }
        }
    }

    MappedFile file;
    ArtifactIdentity identity;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    // v3 logical namespace: fork-side object name -> entries index.  Built from
    // the v3 bindings/uses/components sections; Reader::find consults it when
    // the physical index misses.
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>>
        logical_index;
    // v3 row ranges: binding name -> (object, first row, row count). See ObjectSlice.
    std::unordered_map<std::string, ObjectSlice, TransparentStringHash, std::equal_to<>>
        slice_index;
    std::uint64_t payload_start = 0;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->identity; }

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto it = impl_->index.find(name);
    if (it != impl_->index.end()) { return &impl_->entries[it->second]; }
    const auto logical = impl_->logical_index.find(name);
    return logical == impl_->logical_index.end() ? nullptr
                                                 : &impl_->entries[logical->second];
}

const ObjectSlice* Reader::find_slice(std::string_view name) const noexcept {
    const auto it = impl_->slice_index.find(name);
    return it == impl_->slice_index.end() ? nullptr : &it->second;
}

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file.size(); }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    const auto absolute =
        checked_add(impl_->payload_start, object_offset(object), "absolute payload offset");
    const auto end = checked_add(absolute, object_bytes(object), "absolute payload range");
    if (end > impl_->file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        absolute,
        std::span<const std::byte>(impl_->file.data() + absolute,
                                   static_cast<std::size_t>(object_bytes(object))),
    };
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

std::size_t Reader::read_direct(std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    return impl_->file.read_direct(absolute_offset, destination);
}

} // namespace ninfer::artifact
