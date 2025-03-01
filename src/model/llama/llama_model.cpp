// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llama_model.hpp"

#include "backend/cpu_buffer.hpp"
#include "core/logger.hpp"
#include "executor/executor.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/llama/llama_weight.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace powerserve {

LlamaModel::LlamaModel(const std::string &filename, const std::shared_ptr<ModelConfig> &config) : Model(filename) {
    {
        gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx};
        gguf_ctx                = gguf_init_from_file(filename.c_str(), params);
        POWERSERVE_ASSERT(gguf_ctx != nullptr);
        POWERSERVE_ASSERT(ggml_ctx != nullptr);
    }
    m_config  = config;
    lazy_load = ggml_get_tensor(ggml_ctx, "output_norm.weight") == nullptr ? true : false;
    m_weights = std::make_shared<LlamaWeight>(ggml_ctx, m_config->llm.n_layers, lazy_load);
    if (lazy_load) {
        POWERSERVE_LOG_WARN("only the embedding table was loaded");
    }
    m_ffn = std::make_shared<FFN>(m_config->llm, m_weights);
}

LlamaModel::~LlamaModel() {
    gguf_free(gguf_ctx);
}

auto LlamaModel::forward(
    const std::vector<int> &tokens, const std::vector<int> &pos, const CausalAttentionMask &mask, bool lm_head
) -> LogitsVector {
    Graph g(m_config->model_id);
    // input embedding
    // for (auto t : tokens) {
    //     std::cout << t << " ";
    // }
    // std::cout << std::endl;
    // for (auto p : pos) {
    //     std::cout << p << " ";
    // }
    // std::cout << std::endl;

    size_t batch_size = tokens.size();
    // size_t batch_size  = tokens.size();
    auto embd_tb       = g.add_tensor(m_weights->token_embedding_table);
    auto x             = g.get_embedding(embd_tb, tokens);
    x->m_name          = fmt::format("embedding_{}", pos[0]);
    TensorNode *logits = nullptr;

    auto &llm_config = m_config->llm;

#if defined(POWERSERVE_WITH_QNN)
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
            // TODO:
            static_cast<ggml::GGMLBackend &>(*m_platform->backends[m_config->model_id][TensorBackend::GGML_CPU]).reset_kv_batch_size(batch_size);
            // m_platform->ggml_backends[m_config->model_id]->reset_kv_batch_size(batch_size);
            for (size_t L = 0; L < llm_config.n_layers; L++) {
#if defined(POWERSERVE_WITH_CUDA)
                auto [k_ptr, v_ptr] = static_cast<ggml_cuda::GGML_CUDABackend &>(*m_platform->backends[m_config->model_id][TensorBackend::GGML_GPU]).m_kv->get_cache(L);
                auto &k_cache{*k_ptr}, &v_cache{*v_ptr};
#else
                auto [k_cache, v_cache] = static_cast<ggml::GGMLBackend &>(*m_platform->backends[m_config->model_id][TensorBackend::GGML_CPU]).m_kv->get_cache(L);
#endif
                k_cache.m_name = fmt::format("k_cache_{}", L);
                v_cache.m_name = fmt::format("v_cache_{}", L);
                auto k_node{g.add_tensor(k_cache)};
                auto v_node{g.add_tensor(v_cache)};
                auto attn_o = m_attn->build(g, x, L, k_node, v_node, pos, mask);

                if (L == llm_config.n_layers - 1) {
                    attn_o = g.view(
                        attn_o,
                        {attn_o->m_shape[0], 1, 1, 1},
                        {sizeof(float),
                         sizeof(float) * attn_o->m_shape[0],
                         sizeof(float) * attn_o->m_shape[0],
                         sizeof(float) * attn_o->m_shape[0]},
                        (tokens.size() - 1) * attn_o->m_shape[0] * sizeof(float)
                    );
                }
                attn_o->m_name = fmt::format("attn_o_{}_{}", L, pos[0]);

                auto ffn_o    = m_ffn->build(g, attn_o, L);
                ffn_o->m_name = fmt::format("ffn_o_{}_{}", L, pos[0]);
                x             = ffn_o;
            }
            // TODO: cpu and qnn reuse
            if (lm_head) {
                auto rms_final_w       = g.add_tensor(m_weights->rms_final_weight);
                auto final_rms_norm    = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                final_rms_norm->m_name = fmt::format("final_rms_norm_{}", pos[0]);
                auto output_w          = g.add_tensor(m_weights->output_weight);
                logits                 = g.mat_mul(output_w, final_rms_norm);
                logits->m_name         = fmt::format("logits_{}", pos[0]);
            }
        }
    }

    Executor executor(*m_platform, g);

    executor.shed_op_to_backend();
    executor.split_graph();
    executor.allocate_buffer_with_backend();

    // std::ofstream graph_file("graph_output_cpu.log");
    // executor.print_graph(graph_file);
    // graph_file.close();
    // executor.run();
    executor.run_with_backend();
#if defined(POWERSERVE_WITH_QNN)
    if (!m_platform->qnn_backend)
#endif
    {
        static_cast<ggml::GGMLBackend &>(* m_platform->backends[m_config->model_id][TensorBackend::GGML_CPU]).m_kv->advance(batch_size);
#if defined(POWERSERVE_WITH_CUDA)
        static_cast<ggml_cuda::GGML_CUDABackend &>(* m_platform->backends[m_config->model_id][TensorBackend::GGML_GPU]).m_kv->advanced_kv_cache_size(batch_size);
#endif
    }

    if (!lm_head) {
        return LogitsVector();
    }

    return LogitsVector(logits->m_data, m_config->llm.vocab_size, batch_size);
}

auto LlamaModel::decode(Sampler &sampler, const std::vector<Token> tokens, const std::vector<int> pos, bool lm_head)
    -> std::vector<Token> {
    auto mask = CausalAttentionMask(tokens.size());
    auto ret  = forward(tokens, pos, mask, lm_head);
    std::vector<Token> toks;

    for (auto logits : ret.logits_vector) {
        auto probs = ProbArray(logits);
        sampler.apply(probs);
        auto next = probs.greedy_sample().token;
        sampler.accept(next);
        toks.push_back(next);
    }
    return toks;
}

auto LlamaModel::generate(
    const Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps, size_t batch_size
) -> std::shared_ptr<TokenIterator> {
    return std::make_shared<ModelTokenIterator>(*this, tokenizer, sampler, prompt, steps, batch_size);
}

} // namespace powerserve
