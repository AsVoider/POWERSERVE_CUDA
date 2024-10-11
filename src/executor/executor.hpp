#pragma once

#include "backend/platform.hpp"
#include "graph/graph.hpp"

namespace smart {

struct Executor {
	Platform &platform;
	Graph &graph;

	Executor(Platform &platform_, Graph &graph_) : platform(platform_), graph(graph_) {}

	void allocate_buffers();
	void run();
};

} // namespace smart
