#pragma once

#include "graph/node.hpp"

namespace smart {

struct Graph {
    std::vector<std::shared_ptr<TensorNode>> tensors;
    std::vector<std::shared_ptr<OpNode>> ops;

    auto add_tensor(const Tensor &tensor) -> TensorNode *;
    auto new_tensor(DataType dtype, const Tensor::Shape &shape) -> TensorNode *;
    auto new_op(OpType type) -> OpNode *;
    auto dup_tensor(TensorNode *tensor) -> TensorNode *;

    auto add(TensorNode *a, TensorNode *b) -> TensorNode *;
    auto mat_mul(TensorNode *x, TensorNode *weight) -> TensorNode *;
    auto rms_norm(TensorNode *x, TensorNode *weight) -> TensorNode *;
    auto silu_hadamard(TensorNode *gate, TensorNode *up) -> TensorNode *;
    auto copy(TensorNode *dst, TensorNode *src, size_t off) -> void;

    struct RopeResult {
        TensorNode *q_out;
        TensorNode *k_out;
    };

    auto rope(TensorNode *q, TensorNode *k, TensorNode *pos) -> RopeResult;

    auto softmax(TensorNode *x) -> TensorNode *;
    auto mha(TensorNode *q, TensorNode *key_cache, TensorNode *val_cache, TensorNode *pos, size_t layer_id)
        -> TensorNode *;
};

} // namespace smart
