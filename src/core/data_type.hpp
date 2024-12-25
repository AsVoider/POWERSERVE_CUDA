#pragma once

#include "common/logger.hpp"
#include "ggml.h"

#include <cstddef>

namespace smart {

enum class DataType {
    UNKNOWN,

    FP32,
    FP16,
    INT32,
    INT64,
    GGML_Q4_0,
    GGML_Q8_0,

    COUNT,
};

static size_t get_type_size(DataType dtype) {
    switch (dtype) {
    case DataType::FP32:
        return sizeof(float);
    case DataType::FP16:
        return ggml_type_size(GGML_TYPE_F16);
    case DataType::INT32:
        return sizeof(int32_t);
    case DataType::INT64:
        return sizeof(int64_t);
    case DataType::GGML_Q4_0:
        return ggml_type_size(GGML_TYPE_Q4_0);
    case DataType::GGML_Q8_0:
        return ggml_type_size(GGML_TYPE_Q8_0);
    default:
        SMART_ASSERT(false);
    }
}

static size_t get_block_size(DataType dtype) {
    switch (dtype) {
    case DataType::FP32:
        return ggml_blck_size(GGML_TYPE_F32);
    case DataType::FP16:
        return ggml_blck_size(GGML_TYPE_F16);
    case DataType::INT32:
        return ggml_blck_size(GGML_TYPE_I32);
    case DataType::INT64:
        return ggml_blck_size(GGML_TYPE_I64);
    case DataType::GGML_Q4_0:
        return ggml_blck_size(GGML_TYPE_Q4_0);
    case DataType::GGML_Q8_0:
        return ggml_blck_size(GGML_TYPE_Q8_0);
    default:
        SMART_ASSERT(false);
    }
}

} // namespace smart
