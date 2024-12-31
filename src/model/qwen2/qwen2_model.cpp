#include "qwen2_model.hpp"

#include "backend/cpu_buffer.hpp"
#include "backend/platform.hpp"
#include "common/logger.hpp"
#include "executor/executor.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/qwen2/qwen2_weight.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace smart {

Qwen2Model::Qwen2Model(const std::string &filename, const std::shared_ptr<ModelConfig> &config) : Model(filename) {
    {
        gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx};
        gguf_ctx                = gguf_init_from_file(filename.c_str(), params);
        SMART_ASSERT(gguf_ctx != nullptr);
        SMART_ASSERT(ggml_ctx != nullptr);
    }
    m_config  = config;
    lazy_load = ggml_get_tensor(ggml_ctx, "output_norm.weight") == nullptr ? true : false;
    m_weights = std::make_shared<Qwen2Weight>(ggml_ctx, m_config->llm.n_layers, lazy_load);
    if (lazy_load) {
        SMART_LOG_WARN("only the embedding table was loaded");
    }
    m_ffn = std::make_shared<FFN>(m_config->llm, m_weights);
}

Qwen2Model::~Qwen2Model() {
    gguf_free(gguf_ctx);
}

auto Qwen2Model::forward(
    const std::vector<int> &tokens, const std::vector<int> &pos, const CausalAttentionMask &mask, bool lm_head
) -> std::vector<std::vector<float>> {
    Graph g(m_config->model_id);
    // input embedding
    size_t batch_size  = tokens.size();
    auto embd_tb       = g.add_tensor(m_weights->token_embedding_table);
    auto x             = g.get_embedding(embd_tb, tokens);
    TensorNode *logits = nullptr;

    auto &llm_config = m_config->llm;

#if defined(SMART_WITH_QNN)
    if (m_platform->qnn_backend) {
        auto size            = llm_config.dim;
        bool use_qnn_lm_head = m_platform->qnn_backend->m_models[m_config->model_id]->m_config.lm_heads.size() > 0;
        if (use_qnn_lm_head) {
            size   = llm_config.vocab_size;
            logits = g.qnn_forward(x, pos, mask, size, lm_head);
        } else {
            x = g.qnn_forward(x, pos, mask, size, lm_head);
            if (lm_head) {
                auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
                auto final_rms_norm = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                auto output_w       = g.add_tensor(m_weights->output_weight);
                logits              = g.mat_mul(output_w, final_rms_norm);
            }
        }
    } else
#endif
    {
        if (!lazy_load) {
            m_platform->ggml_backends[m_config->model_id]->reset_kv_batch_size(batch_size);
            for (size_t L = 0; L < llm_config.n_layers; L++) {
                auto [k_cache, v_cache] = m_platform->ggml_backends[m_config->model_id]->m_kv->get_cache(L);
                auto att_o = m_attn->build(g, x, L, g.add_tensor(k_cache), g.add_tensor(v_cache), pos, mask, true);
                auto ffn_o = m_ffn->build(g, att_o, L);
                x          = ffn_o;
            }
            // TODO: cpu and qnn reuse
            if (lm_head) {
                auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
                auto final_rms_norm = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                auto output_w       = g.add_tensor(m_weights->output_weight);
                logits              = g.mat_mul(output_w, final_rms_norm);
            }
        }
    }

    Executor executor(*m_platform, g);
    executor.allocate_buffers();

    executor.run();
#if defined(SMART_WITH_QNN)
    if (!m_platform->qnn_backend)
#endif
    {
        m_platform->ggml_backends[m_config->model_id]->m_kv->advance(batch_size);
    }

    auto res = std::vector<std::vector<float>>();
    if (lm_head) {
        SMART_ASSERT(logits != nullptr);
        float *logits_data = static_cast<float *>(logits->get<CPUBuffer>().m_data);
        for (size_t i = 0; i < batch_size; i++) {
            res.emplace_back(std::vector<float>(
                logits_data + i * llm_config.vocab_size, logits_data + (i + 1) * llm_config.vocab_size
            ));
        }
    }

    return res;
}

auto Qwen2Model::decode(Sampler &sampler, const std::vector<Token> tokens, const std::vector<int> pos, bool lm_head)
    -> std::vector<Token> {
    auto mask = CausalAttentionMask(tokens.size());
    auto ret  = forward(tokens, pos, mask, lm_head);
    std::vector<Token> toks;
    for (auto logits : ret) {
        auto probs = ProbArray(logits);
        sampler.apply(probs);
        std::mt19937 gen(sampler.m_config.seed);
        auto next = probs.sample(gen).index;
        sampler.accept(next);
        toks.push_back(next);
    }
    return toks;
}

auto Qwen2Model::generate(
    Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps, size_t batch_size
) -> Model::TokenRange {
    return Model::TokenRange(*this, tokenizer, sampler, prompt, steps, batch_size);
}

} // namespace smart
