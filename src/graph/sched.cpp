#include "sched.hpp"
#include "common.hpp"
#include "fmt/base.h"
#include "graph/node.hpp"
#include <cassert>
#include <cstdint>
#include <memory>

namespace smart {

void Sched::execute_op(std::shared_ptr<Operator> op, Platform &platform) {
    switch (op->op_type) {
        case OpType::OP_MUL_MAT:
        {
            // prev: x, weight; next: out
            SMART_ASSERT(op->prev.size() == 2 && op->next.size() == 1);
            // SMART_ASSERT(op->prev[0]->node_type == 't' && op->prev[1]->node_type == 't' && op->next[0]->node_type == 't');

            auto src = std::dynamic_pointer_cast<Tensor>(op->prev[0]);
            auto weight = std::dynamic_pointer_cast<Tensor>(op->prev[1]);
            auto dst = std::dynamic_pointer_cast<Tensor>(op->next[0]);
  
            platform.ggml_backend.matmul(dst.get(), weight.get(), src.get()); 
            break;
        }
        case OpType::OP_RMS_NORM:
        {
            // prev: x, weight; next: out
            SMART_ASSERT(op->prev.size() == 2 && op->next.size() == 1);
            // SMART_ASSERT(op->prev[0]->node_type == 't' && op->prev[1]->node_type == 't' && op->next[0]->node_type == 't');

            auto src = std::dynamic_pointer_cast<Tensor>(op->prev[0]);
            auto weight = std::dynamic_pointer_cast<Tensor>(op->prev[1]);
            auto dst = std::dynamic_pointer_cast<Tensor>(op->next[0]);

            platform.ggml_backend.rmsnorm(dst.get(), src.get(), weight.get());
            break;
        }
        case OpType::OP_SOFT_MAX:
        {
            // prev: x, size; next: {}
            SMART_ASSERT(op->prev.size() == 2);
            // SMART_ASSERT(op->prev[0]->node_type == 't' && op->prev[1]->node_type == 't');

            auto x = std::dynamic_pointer_cast<Tensor>(op->prev[0]);
            auto size = std::dynamic_pointer_cast<Tensor>(op->prev[1]);

            platform.ggml_backend.softmax(x.get(), *((int64_t*)size->data));
            break;
        }
        case OpType::OP_ROPE:
        {
            // prev: q, k, pos; next: {rot_q, rot_k} not used
            SMART_ASSERT(op->prev.size() == 3);
            // SMART_ASSERT(op->prev[0]->node_type == 't' && op->prev[1]->node_type == 't' && op->prev[2]->node_type == 't');

            auto q = std::dynamic_pointer_cast<Tensor>(op->prev[0]);
            auto k = std::dynamic_pointer_cast<Tensor>(op->prev[1]);
            auto pos = std::dynamic_pointer_cast<Tensor>(op->prev[2]);

            platform.ggml_backend.rope(q.get(), k.get(), *((int64_t *)pos->container.data()));
            break;
        }
        case OpType::OP_MHA:
        {
            // prev: q, att, key_cache, val_cache, xb, pos, L; next: {att, xb}
            // check
            SMART_ASSERT(op->prev.size() == 7);

            auto q = std::dynamic_pointer_cast<Tensor>(op->prev[0]);
            auto att = std::dynamic_pointer_cast<Tensor>(op->prev[1]);
            auto key_cache = std::dynamic_pointer_cast<Tensor>(op->prev[2]);
            auto val_cache = std::dynamic_pointer_cast<Tensor>(op->prev[3]);
            auto xb = std::dynamic_pointer_cast<Tensor>(op->prev[4]);
            auto pos = std::dynamic_pointer_cast<Tensor>(op->prev[5]);
            auto L = std::dynamic_pointer_cast<Tensor>(op->prev[6]);

            platform.ggml_backend.multihead_attention(q.get(), att.get(), key_cache.get(), val_cache.get(), xb.get(), *((int64_t *)pos->container.data()), *((int64_t *)L->container.data()));
            break;
        }
        case OpType::OP_RES_CONN:
        {
            // prev: x, xb2; next {}
            SMART_ASSERT(op->prev.size() == 2);
            auto x = std::dynamic_pointer_cast<Tensor>(op->prev[0]);
            auto xb2 = std::dynamic_pointer_cast<Tensor>(op->prev[1]);
            platform.ggml_backend.residual_connection(x.get(), xb2.get());
            break;
        }
        case OpType::OP_SILU_HADAMARD:
        {
            // prev hb(gate), hb2(up); next {}
            SMART_ASSERT(op->prev.size() == 2);
            auto hb = std::dynamic_pointer_cast<Tensor>(op->prev[0]);
            auto hb2 = std::dynamic_pointer_cast<Tensor>(op->prev[1]);
            platform.ggml_backend.silu_hadamard(hb.get(), hb2.get());
            break;
        }
        default: 
            break;
    }
}

void Sched::run(Graph &graph, Platform &platform) {
    for (auto node: graph.nodes) {
        // TODO: DAG treverse
        if (node->node_type == 'o') {
            auto op = std::dynamic_pointer_cast<Operator>(node);
            execute_op(op, platform);
        }
    }
}

} // namespace smart