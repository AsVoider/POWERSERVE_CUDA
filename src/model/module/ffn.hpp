#pragma once

#include "graph/graph.hpp"
#include "model/module/module.hpp"
namespace smart {

struct FFN : Module {
	void build_graph(
		Graph &g,
		std::shared_ptr<LlamaConfig> config,
		std::shared_ptr<LlamaWeight> weights,
		std::shared_ptr<LlamaBuffer> buffer,
		int64_t L,
		int64_t pos) override;
};

} // namespace smart
