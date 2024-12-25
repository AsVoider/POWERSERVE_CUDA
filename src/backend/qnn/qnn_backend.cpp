#include "qnn_backend.hpp"

#include "backend/cpu_buffer.hpp"
#include "common/logger.hpp"

namespace smart::qnn {
QNNBackend::QNNBackend(Path libs_path) : m_session(libs_path) {}

void QNNBackend::load_model(const Path &path, const std::shared_ptr<ModelConfig> &model_config) {
    auto &model_id = model_config->model_id;
    SMART_LOG_INFO("Load model {} from {}", model_id, path);
    if (model_config->vision.num_tokens_per_patch) {
        m_models.insert({model_id, std::make_unique<CausalVLM>(path, model_config, m_session)});
    } else {
        m_models.insert({model_id, std::make_unique<CausalLM>(path, model_config, m_session)});
    }
}

void QNNBackend::forward(
    const std::string &model_id,
    const Tensor *dst,
    const Tensor *src,
    const std::vector<int> &pos,
    const CausalAttentionMask &mask
) {
    auto &model           = m_models.at(model_id);
    auto token_embeddings = std::span<const float>((float *)src->get<CPUBuffer>().m_data, src->n_elements());
    auto pos_size_t       = std::vector<size_t>(pos.size());
    std::transform(pos.begin(), pos.end(), pos_size_t.begin(), [](int v) { return size_t(v); });
    auto main_batches = model->split_batch(token_embeddings, pos_size_t, mask);
    float *dst_data_ptr{};
    // TODO: Norm Datatype convert to QNN Datatype
    if (dst->n_elements() > 1) {
        dst_data_ptr = (float *)dst->get<CPUBuffer>().m_data;
    }
    for (size_t i = 0; i < main_batches.size(); i++) {
        auto &batch = main_batches[i];
        batch.forward();
        if (dst->n_elements() > 1) {
            auto vocab_size   = batch.parent.m_model_config->llm.vocab_size;
            size_t batch_size = batch.pos.size();
            size_t dim        = model->m_model_config->llm.dim;

            const float *out_buf = (float *)batch.chunks.back()->output_buffer();
            size_t size          = batch_size * dim * sizeof(float);

            if (batch.lm_head != nullptr) {
                batch.compute_logits();
                out_buf = (float *)batch.lm_head->output_buffer();
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

void QNNBackend::forward(
    const std::string &model_id,
    const Tensor *dst,
    const Tensor *src,
    const std::vector<std::vector<float>> &pixel_values_list,
    const std::vector<std::pair<int, size_t>> &img_infos,
    std::vector<int> &pos,
    const CausalAttentionMask &mask
) {
    auto &model           = static_cast<CausalVLM &>(*(m_models.at(model_id)));
    auto &vision          = model.m_vision;
    auto token_embeddings = std::span<float>((float *)src->get<CPUBuffer>().m_data, src->n_elements());
    auto pos_size_t       = std::vector<size_t>(pos.size());
    if (pos.size() > 2750) //for mmmu test
        return;
    std::transform(pos.begin(), pos.end(), pos_size_t.begin(), [](int v) { return size_t(v); });
    int64_t v_time = 0;

    assert(pixel_values_list.size() == img_infos.size());
    for (size_t img_idx = 0; img_idx < img_infos.size(); img_idx++) {
        auto &pixel_values         = pixel_values_list[img_idx];
        auto num_patch             = img_infos[img_idx].first;
        auto img_offset            = img_infos[img_idx].second;
        size_t pixel_values_offset = 0;

        for (int pidx = 0; pidx < num_patch; pidx++) {
            memcpy(
                vision->input_buffer(),
                (char *)pixel_values.data() + pixel_values_offset,
                vision->m_tensors.at("pixel_values")->size()
            );
            auto t0 = std::chrono::high_resolution_clock::now();
            vision->execute();
            auto t1 = std::chrono::high_resolution_clock::now();
            v_time += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            memcpy(
                (float *)src->get<CPUBuffer>().m_data + img_offset +
                    (pidx * vision->m_tensors.at("image_embeddings")->size() >> 2),
                vision->output_buffer(),
                vision->m_tensors.at("image_embeddings")->size()
            );
            pixel_values_offset += vision->m_tensors.at("pixel_values")->size();
        }
    }

    SMART_LOG_INFO("\nvit time:{} s", v_time / 1000.0);

    auto main_batches = model.split_batch(token_embeddings, pos_size_t, mask);
    float *dst_data_ptr{};
    if (dst->n_elements() > 1) {
        dst_data_ptr = (float *)dst->get<CPUBuffer>().m_data;
    }
    for (size_t i = 0; i < main_batches.size(); i++) {
        auto &batch = main_batches[i];
        batch.forward();
        if (dst->n_elements() > 1) {
            auto vocab_size   = batch.parent.m_model_config->llm.vocab_size;
            size_t batch_size = batch.pos.size();
            size_t dim        = model.m_model_config->llm.dim;

            const float *out_buf = (float *)batch.chunks.back()->output_buffer();
            size_t size          = batch_size * dim * sizeof(float);

            if (batch.lm_head != nullptr) {
                batch.compute_logits();
                out_buf = (float *)batch.lm_head->output_buffer();
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
