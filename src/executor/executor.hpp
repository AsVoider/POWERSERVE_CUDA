#pragma once

#include "backend/platform.hpp"
#include "graph/graph.hpp"

namespace smart {

struct Executor {
public:
    Platform &m_platform;
    Graph &m_graph;

public:
    Executor(Platform &platform, Graph &graph) : m_platform(platform), m_graph(graph) {}

public:
    void allocate_buffers();
    void run();
};

} // namespace smart
