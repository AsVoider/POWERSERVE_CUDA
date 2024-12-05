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
    auto get_embedding(TensorNode *weight, const std::vector<int> &tokens) -> TensorNode *;
    auto add(TensorNode *a, TensorNode *b) -> TensorNode *;
    auto mat_mul(TensorNode *x, TensorNode *weight) -> TensorNode *;
    auto rms_norm(TensorNode *x, TensorNode *weight, float eps) -> TensorNode *;
    auto silu_hadamard(TensorNode *gate, TensorNode *up) -> TensorNode *;
    void copy(TensorNode *dst, TensorNode *src, size_t off);

#if defined(SMART_WITH_QNN)
    auto qnn_forward(TensorNode *x, std::vector<int> pos, const CausalAttentionMask &mask, size_t size, bool lm_head)
        -> TensorNode *;
#endif

    auto rope(TensorNode *src, const std::vector<int> &pos, const RopeConfig &params) -> TensorNode *;

    auto softmax(TensorNode *x) -> TensorNode *;
    auto mha(TensorNode *q, const std::vector<int> &pos, size_t layer_id, uint32_t n_heads) -> TensorNode *;
    void print(TensorNode *x, size_t size);

    auto quest_attention(
        TensorNode *q, const std::vector<int> &pos, size_t layer_id, std::vector<Region> &regions, uint32_t n_heads
    ) -> TensorNode *;
    void cos_sim(TensorNode *src0, TensorNode *src1);
    void add_cache(TensorNode *src, size_t L, const std::vector<int> &pos, size_t head_id, bool is_k);
};

} // namespace smart
