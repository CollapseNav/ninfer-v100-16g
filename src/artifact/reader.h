#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::artifact {

class ArtifactError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class NumericFormat {
    BF16,
    FP32,
    I32,
    Q4G64_F16S,
    Q5G64_F16S,
    Q6G64_F16S,
    W8G32_F16S,
    NVFP4,
    FP8_E4M3FN_ROW_BF16S,
    PTQ1_0_G128,
    PQ2_0_G128,
    // GGUF block-quantized families. The ModelScope Swift-1.5 artifact stores 401 of its 1192
    // tensors in these, so the container layer accepts them independently of whether a kernel
    // exists yet. Storage is the gguf_blocks_v1 layout: 256-weight superblocks (32 for IQ4_NL)
    // exactly as GGUF writes them, so no conversion is involved on read.
    GGUF_Q2_K,
    GGUF_Q4_K,
    GGUF_Q6_K,
    GGUF_IQ1_S,
    GGUF_IQ1_M,
    GGUF_IQ2_XXS,
    GGUF_IQ2_XS,
    GGUF_IQ2_S,
    GGUF_IQ3_XXS,
    GGUF_IQ3_S,
    GGUF_IQ4_XS,
};

enum class StorageLayout {
    ContiguousLeV1,
    RowSplitK128V1,
    BlockScaleK16M128x4V1,
    RowScaleV1,
    GgufBlocksV1,
};

enum class ResourceEncoding {
    RawBytesV1,
};

std::string_view format_name(NumericFormat format) noexcept;
std::string_view layout_name(StorageLayout layout) noexcept;
std::string_view encoding_name(ResourceEncoding encoding) noexcept;

std::uint64_t tensor_alignment(StorageLayout layout) noexcept;
std::uint64_t resource_alignment(ResourceEncoding encoding) noexcept;
std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape);

struct RowSplitGeometry {
    std::uint64_t rows                 = 0;
    std::uint64_t columns              = 0;
    std::uint64_t padded_columns       = 0;
    std::uint64_t group_size           = 0;
    std::uint64_t groups_per_row       = 0;
    std::uint64_t low_bytes_per_group  = 0;
    std::uint64_t high_bytes_per_group = 0;
    std::uint64_t low_plane_bytes      = 0;
    std::uint64_t high_plane_offset    = 0;
    std::uint64_t high_plane_bytes     = 0;
    std::uint64_t scale_plane_offset   = 0;
    std::uint64_t scale_plane_bytes    = 0;
    std::uint64_t encoded_bytes        = 0;
};

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct BlockScaleGeometry {
    std::uint64_t rows                  = 0;
    std::uint64_t columns               = 0;
    std::uint64_t groups_per_row        = 0;
    std::uint64_t k_tiles               = 0;
    std::uint64_t code_plane_bytes      = 0;
    std::uint64_t scale_plane_offset    = 0;
    std::uint64_t scale_plane_bytes     = 0;
    std::uint64_t weight_divisor_offset = 0;
    std::uint64_t encoded_bytes         = 0;
};

BlockScaleGeometry block_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct RowScaleGeometry {
    std::uint64_t rows               = 0;
    std::uint64_t columns            = 0;
    std::uint64_t code_plane_bytes   = 0;
    std::uint64_t scale_plane_offset = 0;
    std::uint64_t scale_plane_bytes  = 0;
    std::uint64_t encoded_bytes      = 0;
};

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct TensorDescriptor {
    std::string name;
    std::vector<std::uint64_t> shape;
    NumericFormat format;
    StorageLayout layout;
    std::uint64_t offset;
    std::uint64_t bytes;
};

struct ResourceDescriptor {
    std::string name;
    ResourceEncoding encoding;
    std::uint64_t offset;
    std::uint64_t bytes;
};

using ObjectDescriptor = std::variant<TensorDescriptor, ResourceDescriptor>;

std::string_view object_name(const ObjectDescriptor& object) noexcept;
std::uint64_t object_offset(const ObjectDescriptor& object) noexcept;
std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept;

struct PayloadSpan {
    std::uint64_t absolute_offset;
    std::span<const std::byte> data;
};

struct ArtifactIdentity {
    std::string model_id;
    std::string weights_id;

    bool operator==(const ArtifactIdentity&) const = default;
};

/**
 * A version-3 binding that names a ROW RANGE of one stored object. The ModelScope Swift-1.5 artifacts
 * use this heavily: attention query and key are two row ranges of one 7168-row object, the GDN q|k|v|z
 * parent is one 16384-row object with z sometimes sliced out, and the MLP gate|up pair is one
 * 34816-row object on some layers.
 *
 * `range` in the v3 directory is an element offset, so first_row is that offset divided by the
 * object's column count. A row slice of a GGUF block matrix is a pointer and row-count adjustment
 * (every row is a whole number of blocks), which is why the reader can expose it without copying.
 * Version-2 artifacts have no such bindings and their map is empty.
 */
struct ObjectSlice {
    const ObjectDescriptor* object = nullptr;
    std::uint64_t first_row        = 0;
    std::uint64_t rows             = 0;
};

class Reader {
public:
    static constexpr std::size_t direct_io_alignment = 4096;

    explicit Reader(const std::filesystem::path& path);
    ~Reader();

    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    const ArtifactIdentity& identity() const noexcept;
    const std::vector<ObjectDescriptor>& objects() const noexcept;
    const ObjectDescriptor* find(std::string_view name) const noexcept;
    // Version 3 only: the row range a binding names, or nullptr when the binding is absent or names
    // a whole object. See ObjectSlice.
    const ObjectSlice* find_slice(std::string_view name) const noexcept;

    std::uint64_t file_bytes() const noexcept;
    std::uint64_t payload_offset() const noexcept;
    PayloadSpan payload(const ObjectDescriptor& object) const;
    PayloadSpan payload(std::string_view name) const;
    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::artifact
