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

    size_t seq_len;
    size_t kv_dim;

    Tensor key_cache;
    Tensor value_cache;
    // ggml::GGMLBackend *backend;
    // FIXME: shrink config to only contain necessary parameters
    const std::shared_ptr<LlamaConfig> &config;

    void add_key_cache(Graph &g, TensorNode *tensor, size_t L, size_t pos);
    void add_value_cache(Graph &g, TensorNode *tensor, size_t L, size_t pos);

    TensorNode *add_key_cache_node(Graph &g);
    TensorNode *add_value_cache_node(Graph &g);

    KVCache(std::shared_ptr<LlamaConfig> &config, size_t seq_len, size_t kv_dim) :
        seq_len(seq_len),
        kv_dim(kv_dim),
        key_cache(
            DataType::FP32, {config->dim * config->n_kv_heads / config->n_heads, config->seq_len, config->n_layers}
        ),
        value_cache(
            DataType::FP32, {config->dim * config->n_kv_heads / config->n_heads, config->seq_len, config->n_layers}
        ),
        config(config) {
        // FIXME: Too Aggressive to allocate memory
        Tensor::Shape shape = {
            (config->dim * config->n_kv_heads) / config->n_heads, config->seq_len, config->n_layers, 1
        };

        ggml::GGMLBackend backend(config); // tmp
        auto kb           = backend.create_buffer<float>(shape);
        auto vb           = backend.create_buffer<float>(shape);
        key_cache.data   = std::move(kb);
        value_cache.data = std::move(vb);
    }

    ~KVCache() {
        key_cache.data.reset();
        value_cache.data.reset();
    }
};

} // namespace smart