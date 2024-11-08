#pragma once

#include "graph/node.hpp"

namespace smart {

struct Graph {
public:
    std::vector<std::shared_ptr<TensorNode>> tensors;
    std::vector<std::shared_ptr<OpNode>> ops;

public:
    auto add_tensor(const Tensor &tensor) -> TensorNode *;
    auto new_tensor(DataType dtype, const Tensor::Shape &shape) -> TensorNode *;
    auto new_op(OpType type) -> OpNode *;
    auto dup_tensor(TensorNode *tensor) -> TensorNode *;
    auto view_tensor(TensorNode *tensor, Tensor::Shape shape) -> TensorViewNode *;

public:
    auto get_embedding(TensorNode *weight, TensorNode *tokens) -> TensorNode *;
    auto add(TensorNode *a, TensorNode *b) -> TensorNode *;
    auto mat_mul(TensorNode *x, TensorNode *weight) -> TensorNode *;
    auto rms_norm(TensorNode *x, TensorNode *weight) -> TensorNode *;
    auto silu_hadamard(TensorNode *gate, TensorNode *up) -> TensorNode *;
    auto copy(TensorNode *dst, TensorNode *src, size_t off) -> void;

    auto rope(TensorNode *src, TensorNode *pos, const RopeConfig &params) -> TensorNode *;

    auto softmax(TensorNode *x) -> TensorNode *;
    auto mha(
        TensorNode *q, TensorNode *key_cache, TensorNode *val_cache, TensorNode *pos, size_t layer_id, uint32_t n_heads
    ) -> TensorNode *;
    auto print(TensorNode *x, size_t size) -> void;

    auto quest_attention(
        TensorNode *q,
        TensorNode *key_cache,
        TensorNode *val_cache,
        TensorNode *pos,
        size_t layer_id,
        std::vector<Region> &regions,
        uint32_t n_heads
    ) -> TensorNode *;
    auto cos_sim(TensorNode *src0, TensorNode *src1) -> void;
};

} // namespace smart
