#pragma once

#include "graph/graph.hpp"
#include "model/common/weights.hpp"

namespace smart {

struct FFN {
private:
    const ModelConfig::LLMConfig &m_config;
    std::shared_ptr<Weight> m_weights;

public:
    FFN(const ModelConfig::LLMConfig &config, std::shared_ptr<Weight> weights) : m_config(config), m_weights(weights) {}

public:
    TensorNode *build(Graph &g, TensorNode *attn_o, int64_t L);
};

} // namespace smart
