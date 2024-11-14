#pragma once

#include "core/config.hpp"
#include "graph/node.hpp"
#include "model/common/weights.hpp"

namespace smart {

struct Attention {

public:
    std::shared_ptr<Config> m_config;
    std::shared_ptr<Weight> m_weights;

public:
    Attention(const std::shared_ptr<Config> &config, const std::shared_ptr<Weight> &weights) :
        m_config(config),
        m_weights(weights) {}

    virtual ~Attention() = default;

public:
    virtual TensorNode *build(
        Graph &g, TensorNode *x, int64_t L, const std::vector<int> &pos, const CausalAttentionMask &mask
    ) = 0;
};

} // namespace smart
