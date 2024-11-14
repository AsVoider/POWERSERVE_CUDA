#include "qnn_interface.hpp"

#include "backend/ggml/buffer.hpp"

namespace smart::qnn {
QNNBackend::QNNBackend(Path working_folder, const std::shared_ptr<smart::Config> &model_config) {
    if (session.get() == nullptr) {
        session = std::make_shared<qnn::Session>(working_folder);
    }
    ++session->m_count;

    m_causalLM = std::make_unique<CausalLM>(working_folder, model_config, *session);
}

QNNBackend::~QNNBackend() {
    m_causalLM.reset(nullptr);
    --session->m_count;
    if (session->m_count == 0) {
        session.reset();
    }
}

void QNNBackend::forward(
    const smart::Tensor *dst, const smart::Tensor *src, const std::vector<int> &pos, const CausalAttentionMask &mask
) {
    auto token_embeddings = std::span<const float>((float *)src->get<smart::ggml::Buffer>().m_data, src->n_elements());
    auto pos_size_t       = std::vector<size_t>(pos.size());
    std::transform(pos.begin(), pos.end(), pos_size_t.begin(), [](int v) { return size_t(v); });
    auto main_batches = m_causalLM->split_batch(token_embeddings, pos_size_t, mask);
    float *dst_data_ptr{};
    if (dst->n_elements() > 1) {
        dst_data_ptr = (float *)dst->get<ggml::Buffer>().m_data;
    }
    for (size_t i = 0; i < main_batches.size(); i++) {
        auto &batch = main_batches[i];
        batch.forward();
        if (dst->n_elements() > 1) {
            batch.compute_logits();
            auto vocab_size = batch.parent.m_model_config->tf_cfg.vocab_size;
            memcpy(dst_data_ptr, batch.lm_head.output_buffer(), batch.pos.size() * vocab_size * sizeof(float));
            dst_data_ptr += batch.pos.size() * vocab_size;
        }
        batch.save_kv();
        batch.advance();
    }
}

} // namespace smart::qnn
