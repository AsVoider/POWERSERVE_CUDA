#pragma once

#include "core/config.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/common/weights.hpp"
#include "model/module/attention.hpp"

#include <cstdlib>

namespace smart {

struct NormAttention : Attention {

public:
    NormAttention(const ModelConfig::LLMConfig &config, std::shared_ptr<Weight> weights) : Attention(config, weights) {}

    ~NormAttention() = default;

public:
    TensorNode *build(Graph &g, TensorNode *x, int64_t L, const std::vector<int> &pos, const CausalAttentionMask &mask)
        override;
};

} // namespace smart
