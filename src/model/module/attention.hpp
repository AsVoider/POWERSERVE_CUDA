#pragma once

#include "graph/graph.hpp"
#include "model/llama/llama_config.hpp"
#include "model/llama/llama_weight.hpp"
namespace smart {

struct Attention {

	std::shared_ptr<LlamaConfig> config;
	std::shared_ptr<LlamaWeight> weights;

	Attention(std::shared_ptr<LlamaConfig> config, std::shared_ptr<LlamaWeight> weights)
		: config(config), weights(weights) {}

	TensorNode *build(Graph &g, TensorNode* x, int64_t L, int64_t pos);
};

} // namespace smart
