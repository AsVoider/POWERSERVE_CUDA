#pragma once

#include "graph/graph.hpp"
#include "model/module/module.hpp"
namespace smart {

class FFN: public Module {
public:
    void build_graph(Graph &g) override;
};

} // namespace smart