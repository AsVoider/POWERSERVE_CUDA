#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>

namespace smart {

// -------------------
static constexpr uint32_t GGUFFILETAG = 0x46554747;  // "GGUF"
static constexpr uint8_t  GGMLMAXDIMS = 4;
static constexpr uint8_t GGUFDEFAULTALIGNMENT = 32;

// -------------------
enum GGUFType {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,       // marks the end of the enum
};
std::string GGUFType2Str(GGUFType const &t);

// NOTE: always add types at the end of the enum to keep backward compatibility
enum GGMLType {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_Q4_0_4_4 = 31,
    GGML_TYPE_Q4_0_4_8 = 32,
    GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    GGML_TYPE_COUNT,
};

enum class GGUFValueType {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,

    _MAX_,
};
std::string GGUFValueType2Str(GGUFValueType const &vt);

GGUFValueType GGUFType2GGUFValueType(GGUFType const &t);
GGUFType GGUFValueType2GGUFType(GGUFValueType const &vt);

struct GGUFValueTypeDescriptor {
    const char*  name;
    int          size;
};

static constexpr uint8_t vtype_descriptors_size = int(GGUFValueType::_MAX_);
extern GGUFValueTypeDescriptor s_vtype_descriptors[];

// -------------------
// file header
struct GGUFHeader {
    uint32_t magic_number;   // magic number to mark gguf file
    uint32_t version;           // must be `3`
    int64_t  tensor_count;      // the number of tensors in the file
    int64_t  metadata_kv_count; // the number of metadata key-value pairs
};

using GGUFStr = std::string;

struct GGUFArray {
    GGUFType type;
    uint64_t n;  // GGUFv2
    void * data;
};

union GGUFValue {
    uint8_t  uint8;
    int8_t   int8;
    uint16_t uint16;
    int16_t  int16;
    uint32_t uint32;
    int32_t  int32;
    float    float32;
    uint64_t uint64;
    int64_t  int64;
    double   float64;
    bool     bool_;

    GGUFStr str;

    GGUFArray arr;

};

std::string GGUFValue2Str(GGUFValue const &v, GGUFType const &t);

struct GGUFKV {
    GGUFStr key;

    GGUFType  type;
    GGUFValue value;
};

struct GGUFTensorInfo {
    GGUFStr name = "";

    uint32_t n_dims = 0;
    uint64_t ne[GGMLMAXDIMS];  // shape

    GGMLType type = GGMLType::GGML_TYPE_F32;

    uint64_t offset = 0; // offset from start of `data`, must be a multiple of `ALIGNMENT`

    // for writing API
    const void * data;
    size_t size;
};

ssize_t get_tensor_size(GGUFTensorInfo &info);

void gguf_tensor_info_sanitize(GGUFTensorInfo *info);
std::string GGUFTensorInfo2Str(GGUFTensorInfo &info);

struct GGUFContext {
    GGUFHeader header;

    GGUFKV          * kv;
    std::unordered_map<std::string, GGUFTensorInfo> infos;

    std::size_t alignment;
    std::size_t offset;    // offset of `data` from beginning of file
    std::size_t size;      // size of `data` in bytes

    //uint8_t * padding;
    int    fd;
    char * data;
};

// ----------------
inline bool is_integer(GGUFValueType vt) noexcept {
    return vt == GGUFValueType::INT8 || vt == GGUFValueType::UINT8 || vt == GGUFValueType::BOOL
        || vt == GGUFValueType::INT16 || vt == GGUFValueType::UINT16 || vt == GGUFValueType::INT32
        || vt == GGUFValueType::UINT32 || vt == GGUFValueType::INT64 || vt == GGUFValueType::UINT64;
}
inline bool is_float(GGUFValueType vt) noexcept {
    return vt == GGUFValueType::FLOAT32 || vt == GGUFValueType::FLOAT64;
}
inline bool is_string(GGUFValueType vt) noexcept {
    return vt == GGUFValueType::STRING;
}
inline bool is_array(GGUFValueType vt) noexcept {
    return vt == GGUFValueType::ARRAY;
}

} // namespace smart
