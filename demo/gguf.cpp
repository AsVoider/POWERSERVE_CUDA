#include "gguf.hpp"
#include "fmt/format.h"
#include <string>
#include <cassert>

namespace smart {
// -------------
// GGUFType
std::string GGUFType2Str(GGUFType const &t){

    switch (t) {
        case GGUFType::GGUF_TYPE_UINT8   : return "UINT8";
        case GGUFType::GGUF_TYPE_INT8    : return "INT8";
        case GGUFType::GGUF_TYPE_UINT16  : return "UINT16";
        case GGUFType::GGUF_TYPE_INT16   : return "INT16";
        case GGUFType::GGUF_TYPE_UINT32  : return "UINT32";
        case GGUFType::GGUF_TYPE_INT32   : return "INT32";
        case GGUFType::GGUF_TYPE_FLOAT32 : return "FLOAT32";
        case GGUFType::GGUF_TYPE_BOOL    : return "BOOL";
        case GGUFType::GGUF_TYPE_STRING  : return "STRING";
        case GGUFType::GGUF_TYPE_ARRAY   : return "ARRAY";
        case GGUFType::GGUF_TYPE_UINT64  : return "UINT64";
        case GGUFType::GGUF_TYPE_INT64   : return "INT64";
        case GGUFType::GGUF_TYPE_FLOAT64 : return "FLOAT64"; 
        default: return fmt::format("UNK id: {}", (int)t);;
    }
    return "UNKNOWN";
}

// -------------
// GGUFValueType
std::string GGUFValueType2Str(GGUFValueType const &vt){

    switch (vt) {
        case GGUFValueType::UINT8   : return "UINT8";
        case GGUFValueType::INT8    : return "INT8";
        case GGUFValueType::UINT16  : return "UINT16";
        case GGUFValueType::INT16   : return "INT16";
        case GGUFValueType::UINT32  : return "UINT32";
        case GGUFValueType::INT32   : return "INT32";
        case GGUFValueType::FLOAT32 : return "FLOAT32";
        case GGUFValueType::BOOL    : return "BOOL";
        case GGUFValueType::STRING  : return "STRING";
        case GGUFValueType::ARRAY   : return "ARRAY";
        case GGUFValueType::UINT64  : return "UINT64";
        case GGUFValueType::INT64   : return "INT64";
        case GGUFValueType::FLOAT64 : return "FLOAT64"; 
        default: return fmt::format("UNK id: {}", (int)vt);
    }
    return "UNKNOWN";
}

GGUFValueType GGUFType2GGUFValueType(GGUFType const &t) {
    // return static_cast<GGUFValueType>(static_cast<std::underlying_type_t<GGUFType>>(t));
    return GGUFValueType(t);
}
GGUFType GGUFValueType2GGUFType(GGUFValueType const &vt) {
    // return static_cast<GGUFType>(static_cast<std::underlying_type_t<GGUFValueType>>(vt));
    return GGUFType(vt);
}

GGUFValueTypeDescriptor s_vtype_descriptors[vtype_descriptors_size] = {
    {"UINT8",   1},  {"INT8",   1},  {"UINT16",  2},  {"INT16",  2},
    {"UINT32",  4},  {"INT32",  4},  {"FLOAT32", 4},  {"BOOL",   1},
    {"STRING", -1},  {"ARRAY", -1},  {"UINT64",  8},  {"INT64",  8},
    {"FLOAT64", 8},
};

// ----------------
// GGUFValue

std::string format_arr(GGUFArray const &arr, uint64_t limit = -1) {
    if (limit < 0 || limit > arr.n)
        limit = arr.n;
    std::string ret = fmt::format("{}[{}] [", GGUFType2Str(arr.type), arr.n);
    
    bool is_fixed = s_vtype_descriptors[int(arr.type)].size > 0;
    if (is_string(GGUFValueType(arr.type))) {
        for (auto i = 0; i < limit; i++) {
            ret = fmt::format("{} {}", ret, ((std::string *)arr.data)[i]);
        }
    } else if (is_fixed) {
        for (auto i = 0; i < limit; i++) {
            switch (arr.type) {
                case GGUFType::GGUF_TYPE_FLOAT32: ret = fmt::format("{} {:.2f}", ret, ((float *) arr.data)[i]); break;
                case GGUFType::GGUF_TYPE_FLOAT64: ret = fmt::format("{} {:.2f}", ret, ((double *) arr.data)[i]); break;
                case GGUFType::GGUF_TYPE_INT8   : ret = fmt::format("{} {}", ret, ((int8_t *) arr.data)[i]); break;
                case GGUFType::GGUF_TYPE_UINT8  : ret = fmt::format("{} {}", ret, ((uint8_t *) arr.data)[i]); break;
                case GGUFType::GGUF_TYPE_INT16  : ret = fmt::format("{} {}", ret, ((int16_t *) arr.data)[i]); break;
                case GGUFType::GGUF_TYPE_UINT16 : ret = fmt::format("{} {}", ret, ((uint16_t *) arr.data)[i]); break;
                case GGUFType::GGUF_TYPE_INT32  : ret = fmt::format("{} {}", ret, ((int32_t *) arr.data)[i]); break;
                case GGUFType::GGUF_TYPE_UINT32 : ret = fmt::format("{} {}", ret, ((uint32_t *) arr.data)[i]); break;
                case GGUFType::GGUF_TYPE_INT64  : ret = fmt::format("{} {}", ret, ((int64_t *) arr.data)[i]); break;
                case GGUFType::GGUF_TYPE_UINT64 : ret = fmt::format("{} {}", ret, ((uint64_t *) arr.data)[i]); break;
                case GGUFType::GGUF_TYPE_BOOL   : ret = fmt::format("{} {}", ret, ((bool *) arr.data)[i]); break;
                default: break;
            }
            ret = fmt::format("{} {}", ret, (char *)arr.data + s_vtype_descriptors[int(arr.type)].size*i);
        }
    }
    ret = fmt::format("{} {}", ret, limit < arr.n ? "... ]" : "]");
    return ret;
}

std::string GGUFValue2Str(GGUFValue const &v, GGUFType const &t) {
    switch (t) {
        case GGUFType::GGUF_TYPE_UINT8   : return fmt::format("{}", v.uint8);
        case GGUFType::GGUF_TYPE_INT8    : return fmt::format("{}", v.int8);
        case GGUFType::GGUF_TYPE_UINT16  : return fmt::format("{}", v.uint16);
        case GGUFType::GGUF_TYPE_INT16   : return fmt::format("{}", v.int16);
        case GGUFType::GGUF_TYPE_UINT32  : return fmt::format("{}", v.uint32);
        case GGUFType::GGUF_TYPE_FLOAT32 : return fmt::format("{}", v.float32);
        case GGUFType::GGUF_TYPE_BOOL    : return fmt::format("{}", v.bool_);
        case GGUFType::GGUF_TYPE_STRING  : return fmt::format("{}", v.str);
        case GGUFType::GGUF_TYPE_ARRAY   : return fmt::format("{}", format_arr(v.arr, 10));
        case GGUFType::GGUF_TYPE_UINT64  : return fmt::format("{}", v.uint64);
        case GGUFType::GGUF_TYPE_INT64   : return fmt::format("{}", v.int64);
        case GGUFType::GGUF_TYPE_FLOAT64 : return fmt::format("{}", v.float64); 
        default: return fmt::format("UNK id: {}", (int)t);
    }
    return "UNKNOWN";
}

void gguf_tensor_info_sanitize(GGUFTensorInfo *info) {
    assert(info->n_dims >= 1 && info->n_dims <= GGMLMAXDIMS);
    assert(0 <= info->type && info->type < GGML_TYPE_COUNT);

    for(uint32_t i = 0; i < info->n_dims; i++) {
        assert(info->ne[i] > 0);
    }
    
    // prevent overflow for total number of elements
    assert(INT64_MAX/info->ne[1] > info->ne[0]);
    assert(INT64_MAX/info->ne[2] > info->ne[0]*info->ne[1]);
    assert(INT64_MAX/info->ne[3] > info->ne[0]*info->ne[1]*info->ne[2]);
}

std::string GGUFTensorInfo2Str(GGUFTensorInfo &info) {
    std::string ret = fmt::format("name: {:25} n_dims: {:2}", info.name, info.n_dims);
    ret = fmt::format("{} ne:({},{},{},{})", ret, info.ne[0], info.ne[1], info.ne[2], info.ne[3]);
    ret = fmt::format("{:64} type:{:2} offset:{:12}", ret, int(info.type), info.offset);
    return ret;
}

ssize_t get_tensor_size(GGUFTensorInfo &info) {
    ssize_t n_elem = 1;
    for (int i = 0; i < info.n_dims; ++i) {
        n_elem *= info.ne[i];
    }
    switch (info.type) {
        case GGML_TYPE_F32: return sizeof(float) * n_elem;
        default: return 0;
    }
    return 0;
}

}
