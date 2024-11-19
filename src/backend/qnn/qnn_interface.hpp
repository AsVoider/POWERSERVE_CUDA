#pragma once

#include "QnnTypes.h"
#include "backend/backend.hpp"
#include "causal_lm.hpp"
#include "common.hpp"
#include "core/config.hpp"
#include "core/data_type.hpp"
#include "core/tensor.hpp"
#include "qnn.hpp"

namespace smart::qnn {

static Qnn_DataType_t convert_datatype_to_qnn(DataType dtp) {
    switch (dtp) {
    case DataType::FP32:
        return QNN_DATATYPE_FLOAT_32;
    default:
        SMART_ASSERT(false);
    }
}

static DataType convert_datatype_from_qnn(Qnn_DataType_t tp) {
    switch (tp) {
    case QNN_DATATYPE_FLOAT_32:
        return DataType::FP32;
    default:
        SMART_ASSERT(false);
    }
}

static std::shared_ptr<qnn::Session> session;

struct QNNBackend : smart::Backend {
    std::unique_ptr<CausalLM> m_causal_lm;

    QNNBackend(Path working_folder, const std::shared_ptr<smart::Config> &model_config);
    virtual ~QNNBackend() override;
    void forward(
        const smart::Tensor *dst, const smart::Tensor *src, const std::vector<int> &pos, const CausalAttentionMask &mask
    );
};

} // namespace smart::qnn
