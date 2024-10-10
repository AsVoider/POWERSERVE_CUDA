#include "graph/graph.hpp"

namespace smart {

auto Graph::add_tensor(const Tensor &tensor) -> TensorNode * {
    return tensors.emplace_back(std::make_shared<TensorNode>(tensor)).get();
}

auto Graph::new_tensor(DataType dtype, Tensor::Shape shape) -> TensorNode * {
    return tensors.emplace_back(std::make_shared<TensorNode>(dtype, shape)).get();
}

auto Graph::new_op(OpType type) -> OpNode * {
    return ops.emplace_back(std::make_shared<OpNode>(type)).get();
}

auto Graph::dup_tensor(TensorNode *tensor) -> TensorNode * {
    return new_tensor(tensor->dtype, tensor->shape);
}

}
