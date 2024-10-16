#pragma once

#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/llama/llama_config.hpp"
#include "model/llama/llama_weight.hpp"
#include "model/module/kv_cache.hpp"

#include <cstdlib>

namespace smart {

struct Attention {
private:
    std::shared_ptr<LlamaConfig> m_config;
    std::shared_ptr<LlamaWeight> m_weights;

private:
    KVCache m_kv_cache;

public:
    Attention(std::shared_ptr<LlamaConfig> config, std::shared_ptr<LlamaWeight> weights) :
        m_config(config),
        m_weights(weights),
        m_kv_cache(config, config->seq_len, config->dim * config->n_kv_heads / config->n_heads) {}

    ~Attention() = default;

public:
    TensorNode *build(Graph &g, TensorNode *x, int64_t L, TensorNode *pos_tensor, int32_t pos);
};

} // namespace smart
