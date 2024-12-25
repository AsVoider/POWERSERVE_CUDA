// 1 platform contains N backends (CPU, NPU, GPU...)
#pragma once

#include "backend/ggml/ggml.hpp"

#if defined(SMART_WITH_QNN)
#include "backend/qnn/qnn_backend.hpp"
#endif

namespace smart {

struct Platform {
    std::unique_ptr<ggml::GGMLBackend> ggml_backend = nullptr;

#if defined(SMART_WITH_QNN)
    std::unique_ptr<qnn::QNNBackend> qnn_backend = nullptr;
#endif

    std::shared_ptr<ModelConfig> m_config = nullptr;

public:
    Platform(std::shared_ptr<ModelConfig> config) : m_config(config) {}

    ~Platform() = default;

public:
    // TODO: No need trans config
    void init_ggml_backend(const std::shared_ptr<ModelConfig> &config, const HyperParams &hparams);

#if defined(SMART_WITH_QNN)
    void init_qnn_backend(const Path &qnn_path);
#endif

    size_t get_kv_position() const;
    void reset_kv_position();
};

} // namespace smart
