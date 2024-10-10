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

auto GraphBuilder::silu_hadamard(TensorNode *gate, TensorNode *up) -> TensorNode * {
    SMART_ASSERT(gate->dtype == up->dtype);
    SMART_ASSERT(gate->shape == up->shape);

    auto out = graph.dup_tensor(gate);
    graph.new_op(OpType::SILU_HADAMARD)
        ->set_inputs({gate, up})
        ->set_outputs({out});

    return out;
}

auto GraphBuilder::rope(TensorNode *q, TensorNode *k, TensorNode *pos)  -> RopeResult {
    // TODO: Add checks

    auto q_out = graph.dup_tensor(q);
    auto k_out = graph.dup_tensor(k);
    graph.new_op(OpType::ROPE)
        ->set_inputs({q, k, pos})
        ->set_outputs({q_out, k_out});

    return {.q_out = q_out, .k_out = k_out};
}

auto GraphBuilder::softmax(TensorNode *x) -> TensorNode * {
    auto out = graph.dup_tensor(x);
    graph.new_op(OpType::SOFTMAX)
        ->set_inputs({x})
        ->set_outputs({out});

    return out;
}

auto GraphBuilder::mha(TensorNode *q, TensorNode *key_cache, TensorNode *val_cache, TensorNode *pos, size_t layer_id) -> TensorNode * {
    // TODO: Add checks

    auto out = graph.dup_tensor(q);
    graph.new_op(OpType::MHA)
        ->set_inputs({q, key_cache, val_cache, pos})
        ->set_outputs({out})
        ->set_params(MHAParams{.layer_id = layer_id});

    return out;
}

}
