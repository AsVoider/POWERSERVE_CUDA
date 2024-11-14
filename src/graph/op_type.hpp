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
#endif

    MHA,
    PRINT,
    QUEST_ATTN,
    COS_SIM,
    GET_EMBEDDING,
    ADD_CACHE,
};

} // namespace smart
