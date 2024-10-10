#pragma once

#include "graph/node.hpp"

namespace smart {

struct Graph {
    std::vector<std::shared_ptr<TensorNode>> tensors;
    std::vector<std::shared_ptr<OpNode>> ops;

    auto add_tensor(const Tensor &tensor) -> TensorNode *;
    auto new_tensor(DataType dtype, Tensor::Shape shape) -> TensorNode *;
    auto new_op(OpType type) -> OpNode *;
    auto dup_tensor(TensorNode *tensor) -> TensorNode *;
};

}
