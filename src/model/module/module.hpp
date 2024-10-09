#pragma once

#include "graph/graph.hpp"
namespace smart {

class Module {
public:

    virtual void build_graph(Graph &g) = 0;
    Module() = default;
    ~Module() = default;

};

} // namespace smart