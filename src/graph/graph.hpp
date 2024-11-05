#pragma once

#include "graph/node.hpp"

namespace smart {

struct Graph {

public:
    struct RopeResult {
        TensorNode *q_out;
        TensorNode *k_out;
    };

public:
    std::vector<std::shared_ptr<TensorNode>> tensors;
    std::vector<std::shared_ptr<OpNode>> ops;

public:
    auto add_tensor(const Tensor &tensor) -> TensorNode *;
    auto new_tensor(DataType dtype, const Tensor::Shape &shape) -> TensorNode *;
    auto new_op(OpType type) -> OpNode *;
    auto dup_tensor(TensorNode *tensor) -> TensorNode *;

public:
    auto add(TensorNode *a, TensorNode *b) -> TensorNode *;
    auto mat_mul(TensorNode *x, TensorNode *weight) -> TensorNode *;
    auto rms_norm(TensorNode *x, TensorNode *weight) -> TensorNode *;
    auto silu_hadamard(TensorNode *gate, TensorNode *up) -> TensorNode *;
    auto copy(TensorNode *dst, TensorNode *src, size_t off) -> void;

    // auto rope(TensorNode *q, TensorNode *k, TensorNode *pos) -> RopeResult;
    auto rope(
        TensorNode *src,
        TensorNode *pos,
        int n_dims = 0,
        int n_ctx_orig = 0,
        float freq_base = 10000.0f,
        float freq_scale = 1.0f,
        float ext_factor = 0.0f,
        float attn_factor = 1.0f,
        float beta_fast = 32.0f,
        float beta_slow = 1.0f
    ) -> TensorNode *;

    auto softmax(TensorNode *x) -> TensorNode *;
    auto mha(TensorNode *q, TensorNode *key_cache, TensorNode *val_cache, TensorNode *pos, size_t layer_id)
        -> TensorNode *;
    auto print(TensorNode *x, size_t size) -> void;

    auto quest_attention(
        TensorNode *q,
        TensorNode *key_cache,
        TensorNode *val_cache,
        TensorNode *pos,
        size_t layer_id,
        std::vector<Region> &regions
    ) -> TensorNode *;
    auto cos_sim(TensorNode *src0, TensorNode *src1) -> void;
};

} // namespace smart
