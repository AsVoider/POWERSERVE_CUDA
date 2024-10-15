#pragma once

#include "backend/ggml/ggml.hpp"
#include "core/tensor.hpp"
#include "fmt/base.h"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/llama/llama_config.hpp"
#include "model/llama/llama_weight.hpp"
#include "model/module/kv_cache.hpp"
#include <cstdlib>
#include <vector>
namespace smart {

struct Attention {

	std::shared_ptr<LlamaConfig> config_;
	std::shared_ptr<LlamaWeight> weights_;

	KVCache kv_cache_;

	Attention(std::shared_ptr<LlamaConfig> config, std::shared_ptr<LlamaWeight> weights)
		: config_(config), weights_(weights), kv_cache_(config, config->seq_len, config->dim * config->n_kv_heads / config->n_heads) {}
	~Attention() = default;

	TensorNode *build(Graph &g, TensorNode *x, int64_t L, TensorNode *pos_tensor, int32_t pos);
};

} // namespace smart
