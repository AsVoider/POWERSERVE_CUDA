// 1 platform contains N backends (CPU, NPU, GPU...)
#pragma once

#include "backend/ggml/ggml.hpp"

#if defined(SMART_WITH_QNN)
#include "backend/qnn/qnn_interface.hpp"
#endif

namespace smart {

struct Platform {
    std::unique_ptr<ggml::GGMLBackend> ggml_backend = nullptr;

#if defined(SMART_WITH_QNN)
    std::unique_ptr<qnn::QNNBackend> qnn_backend = nullptr;
#endif

public:
    void init_ggml_backend(const std::shared_ptr<Config> &config, int n_threads);

#if defined(SMART_WITH_QNN)
    void init_qnn_backend(const Path &qnn_path, const std::shared_ptr<Config> &config);
#endif
};

} // namespace smart
