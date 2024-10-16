#pragma once
#include "backend/ggml/ggml.hpp"
#include "core/buffer.hpp"
#include "core/tensor.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/llama/llama_config.hpp"

#include <cstddef>
#include <cstdlib>
#include <utility>

namespace smart {

struct KVCache {

private:
    size_t seq_len = 0;
    size_t kv_dim  = 0;
    // FIXME: shrink config to only contain necessary parameters
    const std::shared_ptr<LlamaConfig> &m_config = nullptr;

private:
    // TODO: how to set default value
    Tensor m_key_cache;
    Tensor m_value_cache;

public:
    KVCache(std::shared_ptr<LlamaConfig> &config, size_t seq_len, size_t kv_dim) :
        seq_len(seq_len),
        kv_dim(kv_dim),
        m_config(config),
        m_key_cache(
            DataType::FP32, {config->dim * config->n_kv_heads / config->n_heads, config->seq_len, config->n_layers}
        ),
        m_value_cache(
            DataType::FP32, {config->dim * config->n_kv_heads / config->n_heads, config->seq_len, config->n_layers}
        ) {
        // FIXME: Too Aggressive to allocate memory
        Tensor::Shape shape = {
            (config->dim * config->n_kv_heads) / config->n_heads, config->seq_len, config->n_layers, 1
        };

        ggml::GGMLBackend backend(config); // tmp
        auto kb            = backend.create_buffer<float>(shape);
        auto vb            = backend.create_buffer<float>(shape);
        m_key_cache.m_data   = std::move(kb);
        m_value_cache.m_data = std::move(vb);
    }

    ~KVCache() {
        m_key_cache.m_data.reset();
        m_value_cache.m_data.reset();
    }

public:
    void add_key_cache(Graph &g, TensorNode *tensor, size_t L, size_t pos);
    void add_value_cache(Graph &g, TensorNode *tensor, size_t L, size_t pos);

    TensorNode *add_key_cache_node(Graph &g);
    TensorNode *add_value_cache_node(Graph &g);
};

} // namespace smart