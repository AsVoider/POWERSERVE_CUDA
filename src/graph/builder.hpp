#pragma once

#include "graph/node.hpp"

#include <tuple>

namespace smart {

struct Graph;

struct GraphBuilder {
    GraphBuilder(Graph &graph_) : graph(graph_) {}

    auto add(TensorNode *a, TensorNode *b) -> TensorNode *;
    auto mat_mul(TensorNode *x, TensorNode *weight) -> TensorNode *;
    auto rms_norm(TensorNode *x, TensorNode *weight) -> TensorNode *;
    auto silu_hadamard(TensorNode *gate, TensorNode *up) -> TensorNode *;

    struct RopeResult {
        TensorNode *q_out;
        TensorNode *k_out;
    };

    auto rope(TensorNode *q, TensorNode *k, TensorNode *pos)  -> RopeResult;

    auto softmax(TensorNode *x) -> TensorNode *;
    auto mha(TensorNode *q, TensorNode *key_cache, TensorNode *val_cache, TensorNode *pos, size_t layer_id) -> TensorNode *;

private:
    Graph &graph;
};

}
