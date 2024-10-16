#pragma once

#include "backend/platform.hpp"
#include "graph/graph.hpp"

namespace smart {

struct Executor {
    Platform &platform;
    Graph &graph;

    Executor(Platform &platform, Graph &graph) : platform(platform), graph(graph) {}

    void allocate_buffers();
    void run();
};

} // namespace smart
