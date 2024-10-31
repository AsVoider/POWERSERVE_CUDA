#include "executor/executor.hpp"

#include "common.hpp"

namespace smart {

void Executor::allocate_buffers() {
    for (auto tensor : m_graph.tensors) {
        if (tensor->m_data) {
            continue;
        }

        switch (tensor->m_dtype) {
        case DataType::FP32: {
            tensor->m_data = m_platform.ggml_backend.create_buffer<float>(tensor->m_shape);
        } break;

        case DataType::INT32: {
            tensor->m_data = m_platform.ggml_backend.create_buffer<int32_t>(tensor->m_shape);
        } break;

        default:
            SMART_ASSERT(false);
        }
    }
}

void Executor::run() {
    for (auto op : m_graph.ops) {
        switch (op->op) {
        case OpType::ADD: {
            auto a   = op->prev[0]->tensor();
            auto b   = op->prev[1]->tensor();
            auto out = op->output();
            // platform_.ggml_backend.add(out, a, b);
            m_platform.ggml_backend.add(out, a, b);
        } break;

        case OpType::MAT_MUL: {
            auto x      = op->prev[0]->tensor();
            auto weight = op->prev[1]->tensor();
            auto out    = op->output();
            m_platform.ggml_backend.matmul(out, weight, x);
        } break;

        case OpType::RMS_NORM: {
            auto x      = op->prev[0]->tensor();
            auto weight = op->prev[1]->tensor();
            auto out    = op->output();
            m_platform.ggml_backend.rmsnorm(out, x, weight);
        } break;

        case OpType::SILU_HADAMARD: {
            auto gate = op->prev[0]->tensor();
            auto up   = op->prev[1]->tensor();
            auto out  = op->output();
            m_platform.ggml_backend.silu_hadamard(out, gate, up);
        } break;

        case OpType::ROPE: {
            auto q     = op->prev[0]->tensor();
            auto k     = op->prev[1]->tensor();
            auto pos   = op->prev[2]->tensor();
            auto q_out = op->next[0]->tensor();
            auto k_out = op->next[1]->tensor();
            m_platform.ggml_backend.rope(q_out, k_out, q, k, pos);
        } break;

        case OpType::SOFTMAX: {
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            m_platform.ggml_backend.softmax(out, x);
        } break;

        case OpType::MHA: {
            auto q          = op->prev[0]->tensor();
            auto key_cache  = op->prev[1]->tensor();
            auto val_cache  = op->prev[2]->tensor();
            auto pos        = op->prev[3]->tensor();
            auto out        = op->output();
            auto [layer_id] = op->get_params<MHAParams>();
            m_platform.ggml_backend.multihead_attention(out, q, key_cache, val_cache, pos, layer_id);
        } break;

        case OpType::COPY: {
            auto dst = op->prev[0]->tensor();
            auto src = op->prev[1]->tensor();
            auto off = op->get_params<CopyParams>().off;
            m_platform.ggml_backend.copy(dst, src, off);
        } break;

        case OpType::QUEST_ATTN: {
            auto q                   = op->prev[0]->tensor();
            auto key_cache           = op->prev[1]->tensor();
            auto val_cache           = op->prev[2]->tensor();
            auto pos                 = op->prev[3]->tensor();
            auto out                 = op->output();
            auto [layer_id, regions] = op->get_params<QuestAttnParams>();
            m_platform.ggml_backend.quest_attention(out, q, key_cache, val_cache, pos, layer_id, regions);
        } break;

        case OpType::COS_SIM: {
            auto src0 = op->prev[0]->tensor();
            auto src1 = op->prev[1]->tensor();
            m_platform.ggml_backend.cos_sim(src0, src1);
        } break;

        case OpType::PRINT: {
            SMART_ASSERT(false); // not Impl
            // auto x    = op->prev[0]->tensor();
            // auto size = op->get_params<PrintParams>().size;
            // m_platform.ggml_backend.print(x, size);

        } break;

        default:
            SMART_ASSERT(false);
        }
    }
}

} // namespace smart
