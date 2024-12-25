#include "platform.hpp"

namespace smart {

void Platform::init_ggml_backend(const std::shared_ptr<ModelConfig> &config, const HyperParams &hparams) {
    ggml_backends.insert({config->model_id, std::make_unique<ggml::GGMLBackend>(config->llm, hparams)});
}

#if defined(SMART_WITH_QNN)
void Platform::init_qnn_backend(const Path &qnn_path) {
    qnn_backend = std::make_unique<qnn::QNNBackend>(qnn_path);
}
#endif

size_t Platform::get_kv_position(std::string &model_id) const {
    size_t position = ggml_backends.at(model_id)->m_kv->kv_cache->position;
#if defined(SMART_WITH_QNN)
    if (qnn_backend) {
        position = qnn_backend->m_models[model_id]->kv_cache->position;
    }
#endif
    return position;
}

void Platform::reset_kv_position(std::string &model_id) {
    ggml_backends[model_id]->m_kv->reset_kv_cache();
#if defined(SMART_WITH_QNN)
    if (qnn_backend) {
        qnn_backend->m_models[model_id]->reset_kv_cache();
    }
#endif
}

} // namespace smart
