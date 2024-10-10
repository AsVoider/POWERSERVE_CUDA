#pragma once

#include "graph/graph.hpp"
#include "backend/platform.hpp"

namespace smart {

struct Executor {
    void run(const Platform &platform, const Graph &graph);
};

}
