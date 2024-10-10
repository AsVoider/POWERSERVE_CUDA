#pragma once

namespace smart {

enum class DataType {
    UNKNOWN,
    
    FP32,
    FP16,
    GGML_Q4_0,
    GGML_Q8_0,

    COUNT,
};

}
