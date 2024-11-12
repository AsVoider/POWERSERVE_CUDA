#pragma once
#include "backend/ggml/buffer.hpp"
#include "core/buffer.hpp"
#include "core/tensor.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"

#include <cstddef>
#include <cstdlib>
#include <vector>

namespace smart {

struct KVCache {

private:
    size_t m_seq_len  = 0;
    size_t m_kv_dim   = 0;
    size_t m_n_layers = 0;
    std::vector<float> m_key_buf;
    std::vector<float> m_value_buf;

private:
    // TODO: how to set default value
    Tensor m_key_cache;
    Tensor m_value_cache;

public:
    KVCache(size_t seq_len, size_t kv_dim, size_t n_layers) :
        m_seq_len(seq_len),
        m_kv_dim(kv_dim),
        m_n_layers(n_layers),
        m_key_cache(DataType::FP32, {kv_dim, seq_len, n_layers, 1}),
        m_value_cache(DataType::FP32, {kv_dim, seq_len, n_layers, 1}) {

        ggml::Buffer::Stride stride = {
            sizeof(float),
            sizeof(float) * m_seq_len,
            sizeof(float) * m_seq_len * m_n_layers,
            sizeof(float) * m_seq_len * m_n_layers * 1
        };

        m_key_buf.reserve(m_kv_dim * m_seq_len * m_n_layers);
        m_value_buf.reserve(m_kv_dim * m_seq_len * m_n_layers);

        m_key_cache.m_data   = std::make_shared<ggml::Buffer>(stride, m_key_buf.data());
        m_value_cache.m_data = std::make_shared<ggml::Buffer>(stride, m_value_buf.data());
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
