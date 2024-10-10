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

auto Graph::Builder::add(TensorNode *a, TensorNode *b) -> TensorNode * {
    SMART_ASSERT(a->dtype == b->dtype);
    SMART_ASSERT(a->shape == b->shape);

    auto c = graph.new_tensor(a->dtype, a->shape);
    graph.new_op(OpType::ADD)
        ->set_inputs({a, b})
        ->set_outputs({c});

    return c;
}

}
