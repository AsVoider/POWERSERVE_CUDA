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

    MHA,
    PRINT,
    QUEST_ATTN,
    COS_SIM,
    GET_EMBEDDING,
    ADD_CACHE,
    INSERT_IMG_EMBEDDIGN
};

} // namespace smart
