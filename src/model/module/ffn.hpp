#pragma once

#include "graph/graph.hpp"
#include "model/llama/llama_weight.hpp"

namespace smart {

struct FFN {
	std::shared_ptr<LlamaConfig> config;
	std::shared_ptr<LlamaWeight> weights;

	FFN(std::shared_ptr<LlamaConfig> config, std::shared_ptr<LlamaWeight> weights)
		: config(config), weights(weights) {}

	TensorNode *build(Graph &g, TensorNode *attn_o, int64_t L);
};

} // namespace smart
