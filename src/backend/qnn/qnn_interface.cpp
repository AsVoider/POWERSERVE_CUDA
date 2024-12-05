#include "qnn_interface.hpp"

#include "backend/ggml/buffer.hpp"

namespace smart::qnn {
QNNBackend::QNNBackend(Path working_folder, const std::shared_ptr<smart::Config> &model_config) {
    if (session.get() == nullptr) {
        session = std::make_shared<qnn::Session>(working_folder);
    }
    ++session->m_count;

    m_causal_lm = std::make_unique<CausalLM>(working_folder, model_config, *session);
}

QNNBackend::~QNNBackend() {
    m_causal_lm.reset(nullptr);
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
    auto main_batches = m_causal_lm->split_batch(token_embeddings, pos_size_t, mask);
    float *dst_data_ptr{};
    if (dst->n_elements() > 1) {
        dst_data_ptr = (float *)dst->get<ggml::Buffer>().m_data;
    }
    for (size_t i = 0; i < main_batches.size(); i++) {
        auto &batch = main_batches[i];
        batch.forward();
        if (dst->n_elements() > 1) {
            auto vocab_size   = batch.parent.m_model_config->tf_cfg.vocab_size;
            size_t batch_size = batch.pos.size();
            size_t dim        = m_causal_lm->m_model_config->tf_cfg.dim;

            const float *out_buf = batch.chunks.back()->output_buffer();
            size_t size          = batch_size * dim * sizeof(float);

            if (batch.lm_head != nullptr) {
                batch.compute_logits();
                out_buf = batch.lm_head->output_buffer();
                size    = batch_size * vocab_size * sizeof(float);
            }
            memcpy(dst_data_ptr, out_buf, size);
            if (batch.lm_head != nullptr) {
                dst_data_ptr += batch_size * vocab_size;
            }
        }
        batch.save_kv();
        batch.advance();
    }
}

} // namespace smart::qnn
