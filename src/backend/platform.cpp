#include "platform.hpp"

namespace smart {

void Platform::init_ggml_backend(const std::shared_ptr<LLMConfig> &config, int n_threads) {
    ggml_backend = std::make_unique<ggml::GGMLBackend>(config, n_threads);
}

#if defined(SMART_WITH_QNN)
void Platform::init_qnn_backend(const Path &qnn_path, const std::shared_ptr<LLMConfig> &config) {
    qnn_backend = std::make_unique<qnn::QNNBackend>(qnn_path, config);
}
#endif

} // namespace smart
