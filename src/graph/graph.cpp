#include "graph/graph.hpp"

namespace smart {

// Add a tensorNode from existing tensor to graph and this tensor have allocated memory
auto Graph::add_tensor(const Tensor &tensor) -> TensorNode * {
    return tensors.emplace_back(new TensorNode(tensor)).get();
}

// Create a new tensorNode and allocate memory when call Executor::allocate_buffers
auto Graph::new_tensor(DataType dtype, const Tensor::Shape &shape) -> TensorNode * {
    return tensors.emplace_back(new TensorNode(dtype, shape)).get();
}

auto Graph::new_op(OpType type) -> OpNode * {
    return ops.emplace_back(new OpNode(type)).get();
}

// Duplicate a tensorNode(datatype + shape) and **Note**: but not share the same memory
auto Graph::dup_tensor(TensorNode *tensor) -> TensorNode * {
    return new_tensor(tensor->m_dtype, tensor->m_shape);
}

auto Graph::add(TensorNode *a, TensorNode *b) -> TensorNode * {
    SMART_ASSERT(a->m_dtype == b->m_dtype);
    SMART_ASSERT(a->m_shape == b->m_shape);

    auto out = dup_tensor(a);
    auto op  = new_op(OpType::ADD);
    op->set_inputs({a, b});
    op->set_outputs({out});

    return out;
}

auto Graph::mat_mul(TensorNode *x, TensorNode *weight) -> TensorNode * {
    // TODO: Add checks
    SMART_ASSERT(x->m_shape[0] == weight->m_shape[0]);

    auto shape = x->m_shape;
    shape[0]   = weight->n_elements() / weight->m_shape[0];
    auto out   = new_tensor(x->m_dtype, shape);
    auto op    = new_op(OpType::MAT_MUL);
    op->set_inputs({x, weight});
    op->set_outputs({out});

    return out;
}

auto Graph::rms_norm(TensorNode *x, TensorNode *weight) -> TensorNode * {
    SMART_ASSERT(weight->n_dims() == 1);
    SMART_ASSERT(x->m_dtype == weight->m_dtype);
    SMART_ASSERT(x->m_shape[0] == weight->m_shape[0]);

    auto out = dup_tensor(x);
    auto op  = new_op(OpType::RMS_NORM);
    op->set_inputs({x, weight});
    op->set_outputs({out});

    return out;
}

auto Graph::silu_hadamard(TensorNode *gate, TensorNode *up) -> TensorNode * {
    SMART_ASSERT(gate->m_dtype == up->m_dtype);
    SMART_ASSERT(gate->m_shape == up->m_shape);

    auto out = dup_tensor(gate);
    auto op  = new_op(OpType::SILU_HADAMARD);
    op->set_inputs({gate, up});
    op->set_outputs({out});

    return out;
}

// auto Graph::rope(TensorNode *q, TensorNode *k, TensorNode *pos) -> RopeResult {
//     // TODO: Add checks

//     auto q_out = dup_tensor(q);
//     auto k_out = dup_tensor(k);
//     auto op    = new_op(OpType::ROPE);
//     op->set_inputs({q, k, pos});
//     op->set_outputs({q_out, k_out});

//     return {.q_out = q_out, .k_out = k_out};
// }
auto Graph::rope(
    TensorNode *src,
    TensorNode *pos,
    int n_dims,
    int n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow
) -> TensorNode * {
    auto out = dup_tensor(src);
    auto op  = new_op(OpType::ROPE);
    op->set_inputs({src, pos});
    op->set_outputs({out});
    op->set_params(
        RopeParams{n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow}
    );

    return out;
}

auto Graph::softmax(TensorNode *x) -> TensorNode * {
    auto out = dup_tensor(x);
    auto op  = new_op(OpType::SOFTMAX);
    op->set_inputs({x});
    op->set_outputs({out});

    return out;
}

auto Graph::mha(TensorNode *q, TensorNode *key_cache, TensorNode *val_cache, TensorNode *pos, size_t layer_id)
    -> TensorNode * {
    // TODO: Add checks

    auto out = dup_tensor(q);
    auto op  = new_op(OpType::MHA);
    op->set_inputs({q, key_cache, val_cache, pos});
    op->set_outputs({out});
    op->set_params(MHAParams{.layer_id = layer_id});

    return out;
}

auto Graph::copy(TensorNode *dst, TensorNode *src, size_t off) -> void {
    auto op = new_op(OpType::COPY);
    op->set_inputs({dst, src});
    op->set_params(CopyParams{.off = off});
}

auto Graph::print(TensorNode *x, size_t size) -> void {
    auto op = new_op(OpType::PRINT);
    op->set_inputs({x});
    op->set_params(PrintParams{.size = size});
}

auto Graph::quest_attention(
    TensorNode *q,
    TensorNode *key_cache,
    TensorNode *val_cache,
    TensorNode *pos,
    size_t layer_id,
    std::vector<Region> &regions
) -> TensorNode * {
    auto out = dup_tensor(q);
    auto op  = new_op(OpType::QUEST_ATTN);
    op->set_inputs({q, key_cache, val_cache, pos});
    op->set_outputs({out});
    op->set_params(QuestAttnParams(layer_id, regions));

    return out;
}

auto Graph::cos_sim(TensorNode *src0, TensorNode *src1) -> void {
    auto op = new_op(OpType::COS_SIM);
    op->set_inputs({src0, src1});
    op->set_outputs({});
}

} // namespace smart
