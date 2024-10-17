#pragma once

#include "graph/graph.hpp"
#include "model/llama/llama_weight.hpp"

namespace smart {

struct FFN {
private:
    std::shared_ptr<LlamaConfig> m_config;
    std::shared_ptr<LlamaWeight> m_weights;

public:
    FFN(std::shared_ptr<LlamaConfig> config, std::shared_ptr<LlamaWeight> weights) :
        m_config(config),
        m_weights(weights) {}

public:
    TensorNode *build(Graph &g, TensorNode *attn_o, int64_t L);
};

} // namespace smart
