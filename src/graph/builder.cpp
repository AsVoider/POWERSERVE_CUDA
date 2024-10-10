#include "graph/graph.hpp"
#include "graph/builder.hpp"

namespace smart {

auto GraphBuilder::add(TensorNode *a, TensorNode *b) -> TensorNode * {
    SMART_ASSERT(a->dtype == b->dtype);
    SMART_ASSERT(a->shape == b->shape);

    auto out = graph.dup_tensor(a);
    graph.new_op(OpType::ADD)
        ->set_inputs({a, b})
        ->set_outputs({out});

    return out;
}

auto GraphBuilder::mat_mul(TensorNode *x, TensorNode *weight) -> TensorNode * {
    // TODO: Add checks
    SMART_ASSERT(x->shape[0] == weight->shape[0]);

    auto shape = x->shape;
    shape[0] = weight->n_elements() / weight->shape[0];
    auto out = graph.new_tensor(x->dtype, shape);
    graph.new_op(OpType::MAT_MUL)
        ->set_inputs({x, weight})
        ->set_outputs({out});

    return out;
}

auto GraphBuilder::rms_norm(TensorNode *x, TensorNode *weight) -> TensorNode * {
    SMART_ASSERT(weight->n_dims() == 1);
    SMART_ASSERT(x->dtype == weight->dtype);
    SMART_ASSERT(x->shape[0] == weight->shape[0]);

    auto out = graph.dup_tensor(x);
    graph.new_op(OpType::RMS_NORM)
        ->set_inputs({x, weight})
        ->set_outputs({out});

    return out;
}

}
