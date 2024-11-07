#pragma once

#include "core/config.hpp"
#include "graph/graph.hpp"
#include "model/common/weights.hpp"
#include "model/module/attention.hpp"
#include "model/module/region.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace smart {

struct QuestAttention : Attention {
public:
    static constexpr uint32_t dense_layers = 2;

public:
    std::vector<std::vector<Region>> regions_; // (seq_len / regions_tokens, n_layers)

public:
    QuestAttention(std::shared_ptr<Config> config, std::shared_ptr<Weight> weights) : Attention(config, weights) {

        auto kv_dim = (config->tf_cfg.dim * config->tf_cfg.n_kv_heads) / config->tf_cfg.n_heads;
        regions_.resize(config->tf_cfg.n_layers - dense_layers);
        for (uint32_t i = 0; i < config->tf_cfg.n_layers - dense_layers; i++) {
            regions_[i].emplace_back(kv_dim);
        }
    }

    ~QuestAttention() = default;

public:
    TensorNode *build(Graph &g, TensorNode *x, int64_t L, TensorNode *pos_tensor, int32_t pos) override;
};

} // namespace smart
