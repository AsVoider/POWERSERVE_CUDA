#pragma once

namespace smart {

enum class OpType {
    NONE = 0,

    ADD,
    MAT_MUL,
    RMS_NORM,
    SILU_HADAMARD,
    ROPE,
    SOFTMAX,
    COPY,

#if defined(SMART_WITH_QNN)
    QNN_FORWARD,
    QNN_FORWARD_VL,
#endif

    PRINT,
    GET_EMBEDDING,
    ADD_CACHE,
    PERMUTE,
    CONT,
    VIEW,
    SOFTMAX_EXT,
    GET_MASK,
    TRANSPOSE,
    INSERT_IMG_EMBEDDIGN,
};

} // namespace smart
