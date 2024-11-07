#pragma once

#include "core/config.hpp"
#include "graph/node.hpp"
#include "model/common/weights.hpp"
#include "model/module/kv_cache.hpp"

namespace smart {

struct Attention {

public:
    std::shared_ptr<Config> m_config;
    std::shared_ptr<Weight> m_weights;
    KVCache m_kv_cache;

public:
    Attention(std::shared_ptr<Config> config, std::shared_ptr<Weight> weights) :
        m_config(config),
        m_weights(weights),
        m_kv_cache(
            config->tf_cfg.seq_len,
            config->tf_cfg.dim * config->tf_cfg.n_kv_heads / config->tf_cfg.n_heads,
            config->tf_cfg.n_layers
        ) {}

    virtual ~Attention() = default;

public:
    virtual TensorNode *build(Graph &g, TensorNode *x, int64_t L, TensorNode *pos_tensor, int32_t pos) = 0;
};

} // namespace smart