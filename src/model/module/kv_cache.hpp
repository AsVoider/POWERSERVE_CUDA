#pragma once
#include "backend/ggml/ggml.hpp"
#include "core/buffer.hpp"
#include "core/tensor.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/common/config.hpp"

#include <cstddef>
#include <cstdlib>
#include <utility>

namespace smart {

struct KVCache {

private:
    size_t seq_len = 0;
    size_t kv_dim  = 0;
    // FIXME: shrink config to only contain necessary parameters
    const std::shared_ptr<Config> &m_config = nullptr;

private:
    // TODO: how to set default value
    Tensor m_key_cache;
    Tensor m_value_cache;

public:
    KVCache(std::shared_ptr<Config> &config, size_t seq_len, size_t kv_dim) :
        seq_len(seq_len),
        kv_dim(kv_dim),
        m_config(config),
        m_key_cache(
            DataType::FP32,
            {config->tf_cfg.dim * config->tf_cfg.n_kv_heads / config->tf_cfg.n_heads,
             config->tf_cfg.seq_len,
             config->tf_cfg.n_layers}
        ),
        m_value_cache(
            DataType::FP32,
            {config->tf_cfg.dim * config->tf_cfg.n_kv_heads / config->tf_cfg.n_heads,
             config->tf_cfg.seq_len,
             config->tf_cfg.n_layers}
        ) {
        // FIXME: Too Aggressive to allocate memory
        Tensor::Shape shape = {
            (m_config->tf_cfg.dim * m_config->tf_cfg.n_kv_heads) / m_config->tf_cfg.n_heads,
            m_config->tf_cfg.seq_len,
            m_config->tf_cfg.n_layers,
            1
        };

        ggml::GGMLBackend backend(m_config); // tmp
        auto kb              = backend.create_buffer<float>(shape);
        auto vb              = backend.create_buffer<float>(shape);
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