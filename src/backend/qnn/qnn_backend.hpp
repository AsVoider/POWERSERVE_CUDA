#pragma once

#include "QnnTypes.h"
#include "backend/backend.hpp"
#include "causal_models.hpp"
#include "common/type_def.hpp"
#include "core/config.hpp"
#include "core/data_type.hpp"
#include "core/tensor.hpp"
#include "graph_interface.hpp"
#include "qnn.hpp"

namespace smart::qnn {
struct QNNBackend : smart::Backend {
    Session m_session;
    std::map<std::string, std::unique_ptr<CausalLM>> m_models;
    std::map<std::string, std::unique_ptr<Vision>> m_visions;

    QNNBackend(Path libs_path);
    virtual ~QNNBackend() override = default;

    void load_model(const Path &path, const std::shared_ptr<smart::ModelConfig> &model_config);
    void forward(
        const std::string &model_id,
        const smart::Tensor *dst,
        const smart::Tensor *src,
        const std::vector<int> &pos,
        const CausalAttentionMask &mask
    );
    void forward(
        const std::string &model_id,
        const smart::Tensor *dst,
        const smart::Tensor *src,
        const std::vector<std::vector<float>> &pixel_values_list,
        const std::vector<std::pair<int, size_t>> &img_infos,
        std::vector<int> &pos,
        const CausalAttentionMask &mask
    );
};

} // namespace smart::qnn
