#pragma once

#include "core/config.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/common/weights.hpp"
#include "model/module/attention.hpp"

#include <cstdlib>

namespace powerserve {

class FlashAttention : public Attention {

public:
    FlashAttention(const ModelConfig::LLMConfig &config, std::shared_ptr<Weight> weights) :
        Attention(config, weights) {}

    ~FlashAttention() = default;

public:
    TensorNode *build(
        Graph &g,
        TensorNode *x,
        int64_t L,
        const TensorNode *k_cache,
        const TensorNode *v_cache,
        const std::vector<int> &pos,
        const CausalAttentionMask &mask,
        bool is_need_bias = false
    ) override;
};

} // namespace powerserve
