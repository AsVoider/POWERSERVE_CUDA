#include "platform.hpp"

namespace smart {

void Platform::init_ggml_backend(const std::shared_ptr<ModelConfig> &config, const HyperParams &hparams) {
    ggml_backend = std::make_unique<ggml::GGMLBackend>(config->llm, hparams);
}

#if defined(SMART_WITH_QNN)
void Platform::init_qnn_backend(const Path &qnn_path) {
    qnn_backend = std::make_unique<qnn::QNNBackend>(qnn_path);
}
#endif

size_t Platform::get_kv_position() const {
    size_t position = ggml_backend->m_kv->kv_cache->position;
#if defined(SMART_WITH_QNN)
    if (qnn_backend) {
        position = qnn_backend->m_models[m_config->model_id]->kv_cache->position;
    }
#endif
    return position;
}

void Platform::reset_kv_position() {
    ggml_backend->m_kv->reset_kv_cache();
#if defined(SMART_WITH_QNN)
    if (qnn_backend) {
        qnn_backend->m_models[m_config->model_id]->reset_kv_cache();
        // qnn_backend->m_models[m_config->model_id]->kv_cache->truncate(
        //     qnn_backend->m_models[m_config->model_id]->largest_chunks()[0]->m_config.kv_size
        // );
    }
#endif
}

} // namespace smart
