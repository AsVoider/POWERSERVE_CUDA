#pragma once

#include "graph/graph.hpp"
#include "model/llama-impl/llama_buffer.hpp"
#include "model/llama-impl/llama_config.hpp"
#include "model/llama-impl/llama_weight.hpp"
namespace smart {

struct Module {
	virtual void build_graph(
		Graph &g,
		std::shared_ptr<LlamaConfig> config,
		std::shared_ptr<LlamaWeight> weights,
		std::shared_ptr<LlamaBuffer> buffer,
		int64_t L,
		int64_t pos) = 0;
	Module()		 = default;
	~Module()		 = default;
};

} // namespace smart
