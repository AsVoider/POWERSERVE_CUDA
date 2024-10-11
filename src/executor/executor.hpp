#pragma once

#include "backend/platform.hpp"
#include "graph/graph.hpp"

namespace smart {

struct Executor {
	Platform &platform_;
	Graph &graph_;

	Executor(Platform &platform, Graph &graph) : platform_(platform), graph_(graph) {}

	void allocate_buffers();
	void run();
};

} // namespace smart
