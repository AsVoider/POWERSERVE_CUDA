#pragma once

#include "graph/node.hpp"

namespace smart {

struct Graph {
    std::vector<std::shared_ptr<TensorNode>> tensors;
    std::vector<std::shared_ptr<OpNode>> ops;

    auto add_tensor(const Tensor &tensor) -> TensorNode *;
    auto new_tensor(DataType dtype, Tensor::Shape shape) -> TensorNode *;
    auto new_op(OpType type) -> OpNode *;

    struct Builder {
        Graph &graph;

        Builder(Graph &graph_) : graph(graph_) {}

        auto add(TensorNode *a, TensorNode *b) -> TensorNode *;
    };

    Builder builder;

    Graph() : builder(*this) {}
};

}
