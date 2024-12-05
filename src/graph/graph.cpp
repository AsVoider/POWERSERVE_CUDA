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

auto Graph::view_tensor(TensorNode *tensor, Tensor::Shape shape) -> TensorViewNode * {
    return static_cast<TensorViewNode *>(tensors.emplace_back(new TensorViewNode(*tensor, shape)).get());
}

auto Graph::get_embedding(TensorNode *weight, const std::vector<int> &tokens) -> TensorNode * {
    auto op         = new_op(OpType::GET_EMBEDDING);
    auto out        = dup_tensor(weight); // weights (dim, vocab_size)
    out->m_dtype    = DataType::FP32;
    out->m_shape[1] = tokens.size(); // batch size   // inp (dim, batch_size)
    op->set_inputs({weight});
    op->set_outputs({out});
    op->set_params(GetEmbeddingParams{tokens});
    return out;
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

auto Graph::rms_norm(TensorNode *x, TensorNode *weight, float eps) -> TensorNode * {
    SMART_ASSERT(weight->n_dims() == 1);
    SMART_ASSERT(x->m_dtype == weight->m_dtype);
    SMART_ASSERT(x->m_shape[0] == weight->m_shape[0]);

    auto out = dup_tensor(x);
    auto op  = new_op(OpType::RMS_NORM);
    op->set_inputs({x, weight});
    op->set_outputs({out});
    op->set_params(RMSNormParams{.eps = eps});

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

auto Graph::rope(TensorNode *src, const std::vector<int> &pos, const RopeConfig &params) -> TensorNode * {
    auto out = dup_tensor(src);
    auto op  = new_op(OpType::ROPE);
    op->set_inputs({src});
    op->set_outputs({out});
    op->set_params(RopeParams{pos, params});

    return out;
}

auto Graph::softmax(TensorNode *x) -> TensorNode * {
    auto out = dup_tensor(x);
    auto op  = new_op(OpType::SOFTMAX);
    op->set_inputs({x});
    op->set_outputs({out});

    return out;
}

auto Graph::mha(TensorNode *q, const std::vector<int> &pos, size_t layer_id, uint32_t n_heads) -> TensorNode * {
    // TODO: Add checks

    auto out = dup_tensor(q);
    auto op  = new_op(OpType::MHA);
    op->set_inputs({q});
    op->set_outputs({out});
    op->set_params(MHAParams{.pos = pos, .layer_id = layer_id, .n_heads = n_heads});

    return out;
}

#if defined(SMART_WITH_QNN)
auto Graph::qnn_forward(TensorNode *x, std::vector<int> pos, const CausalAttentionMask &mask, size_t size, bool lm_head)
    -> TensorNode * {
    TensorNode *out = nullptr;
    auto op         = new_op(OpType::QNN_FORWARD);
    op->set_inputs({x});
    op->set_params(QNNForwardParams(pos, mask));
    if (lm_head) {
        out = new_tensor(DataType::FP32, {size, x->m_shape[1]}); // size can be vocab_size or dim
    } else {
        out = new_tensor(DataType::FP32, {0});
    }
    op->set_outputs({out});
    return out;
}
#endif

void Graph::copy(TensorNode *dst, TensorNode *src, size_t off) {
    auto op = new_op(OpType::COPY);
    op->set_inputs({dst, src});
    op->set_params(CopyParams{.off = off});
}

void Graph::print(TensorNode *x, size_t size) {
    auto op = new_op(OpType::PRINT);
    op->set_inputs({x});
    op->set_params(PrintParams{.size = size});
}

auto Graph::quest_attention(
    TensorNode *q, const std::vector<int> &pos, size_t layer_id, std::vector<Region> &regions, uint32_t n_heads
) -> TensorNode * {
    auto out = dup_tensor(q);
    auto op  = new_op(OpType::QUEST_ATTN);
    op->set_inputs({q});
    op->set_outputs({out});
    op->set_params(QuestAttnParams{.pos = pos, .layer_id = layer_id, .regions = regions, .n_heads = n_heads});

    return out;
}

void Graph::cos_sim(TensorNode *src0, TensorNode *src1) {
    auto op = new_op(OpType::COS_SIM);
    op->set_inputs({src0, src1});
    op->set_outputs({});
}

void Graph::add_cache(TensorNode *src, size_t L, const std::vector<int> &pos, size_t head_id, bool is_k) {
    auto op = new_op(OpType::ADD_CACHE);
    op->set_inputs({src});
    op->set_params(AddCacheParams{L, pos, head_id, is_k});
}

} // namespace smart
