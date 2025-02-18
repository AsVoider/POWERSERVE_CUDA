// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "graph/graph.hpp"

namespace powerserve {

// Add a tensorNode from existing tensor to graph and this tensor have allocated memory
auto Graph::add_tensor(const Tensor &tensor) -> TensorNode * {
    return tensors.emplace_back(new TensorNode(tensor)).get();
}

// Create a new tensorNode and allocate memory when call Executor::allocate_buffers
auto Graph::new_tensor(DataType dtype, const Shape &shape) -> TensorNode * {
    return tensors.emplace_back(new TensorNode(dtype, shape)).get();
}

auto Graph::new_op(OpType type) -> OpNode * {
    return ops.emplace_back(new OpNode(type)).get();
}

// Duplicate a tensorNode(datatype + shape) and **Note**: but not share the same memory
auto Graph::dup_tensor(TensorNode *tensor) -> TensorNode * {
    return new_tensor(tensor->m_dtype, tensor->m_shape);
}

auto Graph::view_tensor(const TensorNode *tensor, Shape shape) -> TensorViewNode * {
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
#if defined(POWERSERVE_WITH_CUDA)
    out->m_backend = TensorBackend::GGML_GPU;
#else
    out->m_backend = TensorBackend::GGML_CPU;
#endif
    return out;
}

auto Graph::add(TensorNode *a, TensorNode *b) -> TensorNode * {
    POWERSERVE_ASSERT(tensor_can_repeat(b, a));

    auto out = dup_tensor(a);
    out->m_name = "add_out";
    auto op  = new_op(OpType::ADD);
    op->set_inputs({a, b});
    op->set_outputs({out});

    {
        POWERSERVE_ASSERT(a->m_backend == b->m_backend);
        out->m_backend = a->m_backend;        
    }
    return out;
}

auto Graph::mat_mul(TensorNode *a, TensorNode *b) -> TensorNode * {
    // TODO: Add checks
    POWERSERVE_ASSERT(a->m_shape[0] == b->m_shape[0]);
    POWERSERVE_ASSERT(tensor_can_mul_mat(a, b));

    Shape shape = {a->m_shape[1], b->m_shape[1], b->m_shape[2], b->m_shape[3]};
    auto out    = new_tensor(DataType::FP32, shape);
    out->m_name = "mat_mul_out";
    auto op     = new_op(OpType::MAT_MUL);
    op->set_inputs({a, b});
    op->set_outputs({out});

    {
        POWERSERVE_ASSERT(a->m_backend == b->m_backend);
        out->m_backend = a->m_backend;
    }
    return out;
}

auto Graph::rms_norm(TensorNode *x, TensorNode *weight, float eps) -> TensorNode * {
    POWERSERVE_ASSERT(weight->n_dims() == 1);
    POWERSERVE_ASSERT(x->m_dtype == weight->m_dtype);
    POWERSERVE_ASSERT(x->m_shape[0] == weight->m_shape[0]);

    auto out = dup_tensor(x);
    out->m_name = "rms_norm_out";
    auto op  = new_op(OpType::RMS_NORM);
    op->set_inputs({x, weight});
    op->set_outputs({out});
    op->set_params(RMSNormParams{.eps = eps});

    {
        out->m_backend = x->m_backend;
        POWERSERVE_ASSERT(weight == nullptr or weight->m_backend == x->m_backend);
    }
    return out;
}

auto Graph::silu_hadamard(TensorNode *gate, TensorNode *up) -> TensorNode * {
    POWERSERVE_ASSERT(gate->m_dtype == up->m_dtype);
    POWERSERVE_ASSERT(gate->m_shape == up->m_shape);

    auto out = dup_tensor(gate);
    auto op  = new_op(OpType::SILU_HADAMARD);
    op->set_inputs({gate, up});
    op->set_outputs({out});

    {
        POWERSERVE_ASSERT(gate->m_backend == up->m_backend);
        out->m_backend = gate->m_backend;
    }
    return out;
}

auto Graph::rope(TensorNode *src, TensorNode *rope_factors, const std::vector<int> &pos, const ModelConfig::LLMConfig::RopeConfig &params)
    -> TensorNode * {
    // TODO: Only support linear ROPE now
    auto out = dup_tensor(src);
    out->m_name = "rope_out";
    auto op  = new_op(OpType::ROPE);
    op->set_inputs({src, rope_factors});
    op->set_outputs({out});
    op->set_params(RopeParams{pos, params});

    {
        out->m_backend = src->m_backend;
        POWERSERVE_ASSERT(rope_factors == nullptr or rope_factors->m_backend == src->m_backend);
    }
    return out;
}

auto Graph::softmax(TensorNode *x) -> TensorNode * {
    auto out = dup_tensor(x);
    auto op  = new_op(OpType::SOFTMAX);
    op->set_inputs({x});
    op->set_outputs({out});
    {
        out->m_backend = x->m_backend;
    }

    return out;
}

#if defined(POWERSERVE_WITH_QNN)
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

auto Graph::qnn_forward_vl(
    TensorNode *x,
    std::vector<int> pos,
    const CausalAttentionMask &mask,
    size_t vocab_size,
    bool lm_head,
    std::vector<std::vector<float>> &pixel_values_list,
    std::vector<std::pair<int, size_t>> &img_infos
) -> TensorNode * {
    TensorNode *out = nullptr;
    auto op         = new_op(OpType::QNN_FORWARD_VL);
    op->set_inputs({x});
    op->set_params(QNNForwardVLParams(pos, mask, pixel_values_list, img_infos));
    if (lm_head) {
        out = new_tensor(DataType::FP32, {vocab_size, x->m_shape[1]});
    } else {
        out = new_tensor(DataType::FP32, {0});
    }
    op->set_outputs({out});
    return out;
}
#endif

void Graph::copy(TensorNode *dst, TensorNode *src) {
    auto op = new_op(OpType::COPY);
    op->set_inputs({dst, src});
    op->set_params(CopyParams{});
    {
        POWERSERVE_ASSERT(dst->m_backend == src->m_backend);
    }
}

void Graph::print(TensorNode *x, size_t size) {
    auto op = new_op(OpType::PRINT);
    op->set_inputs({x});
    op->set_params(PrintParams{.size = size});
}

void Graph::add_cache(TensorNode *k, TensorNode *v, size_t L, const std::vector<int> &pos, size_t head_id) {
    auto op = new_op(OpType::ADD_CACHE);
    op->set_inputs({k, v});
    op->set_params(AddCacheParams{L, pos, head_id});
    {
        POWERSERVE_ASSERT(k->m_backend == v->m_backend);
    }
}

auto Graph::permute(TensorNode *x, Shape axes) -> TensorViewNode * {
    POWERSERVE_ASSERT(axes[0] < max_n_dims);
    POWERSERVE_ASSERT(axes[1] < max_n_dims);
    POWERSERVE_ASSERT(axes[2] < max_n_dims);
    POWERSERVE_ASSERT(axes[3] < max_n_dims);

    POWERSERVE_ASSERT(axes[0] != axes[1]);
    POWERSERVE_ASSERT(axes[0] != axes[2]);
    POWERSERVE_ASSERT(axes[0] != axes[3]);
    POWERSERVE_ASSERT(axes[1] != axes[2]);
    POWERSERVE_ASSERT(axes[1] != axes[3]);
    POWERSERVE_ASSERT(axes[2] != axes[3]);

    Shape shape{};
    shape[axes[0]] = x->m_shape[0];
    shape[axes[1]] = x->m_shape[1];
    shape[axes[2]] = x->m_shape[2];
    shape[axes[3]] = x->m_shape[3];

    auto out = view_tensor(x, shape);
    out->m_name = "permute_out";
    auto op  = new_op(OpType::PERMUTE);
    op->set_inputs({x});
    op->set_outputs({out});
    op->set_params(PermuteParams{.axes = axes});

    {
        out->m_backend = x->m_backend;
    }
    return out;
}

auto Graph::cont(TensorNode *x, Shape shape) -> TensorNode * {
    auto out = new_tensor(x->m_dtype, shape);
    auto op  = new_op(OpType::CONT);
    op->set_inputs({x});
    op->set_outputs({out});
    op->set_params(ContParams({}));

    {
        out->m_backend = x->m_backend;
    }
    return out;
}

auto Graph::view(const TensorNode *x, Shape shape, Shape stride, size_t offset) -> TensorViewNode * {
    auto out = view_tensor(x, shape);
    auto op  = new_op(OpType::VIEW);
    op->set_inputs({});
    op->set_outputs({out});
    op->set_params(ViewParams({.stride = stride, .offset = offset}));

    {
        out->m_backend = x->m_backend;
    }
    return out;
}

auto Graph::softmax_ext(TensorNode *x, TensorNode *mask, float scale, float max_bias) -> TensorNode * {
    auto out = dup_tensor(x);
    auto op  = new_op(OpType::SOFTMAX_EXT);
    op->set_inputs({x, mask});
    op->set_outputs({out});
    op->set_params(SoftmaxExtParams({.scale = scale, .max_bias = max_bias}));

    {
        POWERSERVE_ASSERT(x->m_backend == mask->m_backend);
        out->m_backend = x->m_backend;
    }
    return out;
}

auto Graph::get_mask(const CausalAttentionMask &mask, Shape shape, const std::vector<int> &pos, TensorNode *kq) -> TensorNode * {
    auto out = new_tensor(DataType::FP32, shape);
    auto op  = new_op(OpType::GET_MASK);
    op->set_outputs({out});
    op->set_params(GetMaskParams{.mask = mask, .pos = pos});

    {
        out->m_backend = kq == nullptr ? TensorBackend::GGML_CPU : kq->m_backend;
    }
    return out;
}

auto Graph::transpose(TensorNode *x) -> TensorViewNode * {
    auto shape{x->m_shape};
    shape[0] = x->m_shape[1];
    shape[1] = x->m_shape[0];

    auto out = view_tensor(x, shape);
    out->m_name = "transpose_out";
    auto op  = new_op(OpType::TRANSPOSE);
    op->set_inputs({x});
    op->set_outputs({out});

    {
        out->m_backend = x->m_backend;
    }
    return out;
}

auto Graph::make_contiguous(TensorNode *x) -> TensorNode * {
    auto out = dup_tensor(x);
    copy(out, x);
    return out;
}

} // namespace powerserve
