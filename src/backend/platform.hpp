// 1 platform contains N backends (CPU, NPU, GPU...)
#pragma once

#include "backend/ggml/ggml.hpp"

#include <map>

#if defined(SMART_WITH_QNN)
#include "backend/qnn/qnn_backend.hpp"
#endif

namespace smart {

struct Platform {
    std::map<std::string, std::unique_ptr<ggml::GGMLBackend>> ggml_backends;

#if defined(SMART_WITH_QNN)
    std::unique_ptr<qnn::QNNBackend> qnn_backend = nullptr;
#endif

public:
    Platform() = default;

    ~Platform() = default;

public:
    // TODO: No need trans config
    void init_ggml_backend(const std::shared_ptr<ModelConfig> &config, const HyperParams &hparams);

#if defined(SMART_WITH_QNN)
    void init_qnn_backend(const Path &qnn_path);
#endif

    size_t get_kv_position(std::string &model_id) const;
    void reset_kv_position(std::string &model_id);
};

} // namespace smart
