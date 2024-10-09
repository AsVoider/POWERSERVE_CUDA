#pragma once

#include "graph/graph.hpp"
#include "model/module/module.hpp"
namespace smart {

class Attention: public Module {
public:
    void build_graph(Graph &g) override;
};

} // namespace smart