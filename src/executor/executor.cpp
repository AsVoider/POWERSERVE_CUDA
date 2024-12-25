#include "executor/executor.hpp"

#include "common/logger.hpp"

#include <cstdint>

namespace smart {

void Executor::allocate_buffers() {
    for (auto tensor : m_graph.tensors) {
        if (tensor->m_data) {
            continue;
        }

        switch (tensor->m_dtype) {
        case DataType::FP32: {
            create_ggml_buffer<float>(tensor);
        } break;

        case DataType::INT32: {
            create_ggml_buffer<int32_t>(tensor);
        } break;
        case DataType::INT64: {
            tensor->m_data = m_platform.ggml_backend->create_buffer<int64_t>(tensor->m_shape);
        } break;

        default:
            SMART_ABORT("could not allocate buffer for data type: {}", static_cast<int>(tensor->m_dtype));
        }
    }
}

void Executor::run() {
    for (auto op : m_graph.ops) {
        switch (op->op) {
        case OpType::GET_EMBEDDING: {
            auto weight   = op->prev[0]->tensor();
            auto out      = op->output();
            auto [tokens] = op->get_params<GetEmbeddingParams>();
            m_platform.ggml_backend->get_embedding(out, weight, tokens);
        } break;

        case OpType::ADD: {
            auto a   = op->prev[0]->tensor();
            auto b   = op->prev[1]->tensor();
            auto out = op->output();
            m_platform.ggml_backend->add(out, a, b);
        } break;

        case OpType::MAT_MUL: {
            auto x      = op->prev[0]->tensor();
            auto weight = op->prev[1]->tensor();
            auto out    = op->output();
            m_platform.ggml_backend->matmul(out, weight, x);
        } break;

        case OpType::RMS_NORM: {
            auto x      = op->prev[0]->tensor();
            auto weight = op->prev[1]->tensor();
            auto out    = op->output();
            auto [eps]  = op->get_params<RMSNormParams>();
            m_platform.ggml_backend->rmsnorm(out, x, weight, eps);
        } break;

        case OpType::SILU_HADAMARD: {
            auto gate = op->prev[0]->tensor();
            auto up   = op->prev[1]->tensor();
            auto out  = op->output();
            m_platform.ggml_backend->silu_hadamard(out, gate, up);
        } break;

        case OpType::ROPE: {
            auto src             = op->prev[0]->tensor();
            auto out             = op->next[0]->tensor();
            auto [pos, rope_cfg] = op->get_params<RopeParams>();
            m_platform.ggml_backend->rope(out, src, pos, rope_cfg);
        } break;

        case OpType::SOFTMAX: {
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            m_platform.ggml_backend->softmax(out, x);
        } break;

        case OpType::MHA: {
            auto q                        = op->prev[0]->tensor();
            auto out                      = op->output();
            auto [pos, layer_id, n_heads] = op->get_params<MHAParams>();
            m_platform.ggml_backend->multihead_attention(out, q, pos, layer_id, n_heads);
        } break;

        case OpType::COPY: {
            auto dst = op->prev[0]->tensor();
            auto src = op->prev[1]->tensor();
            auto off = op->get_params<CopyParams>().off;
            m_platform.ggml_backend->copy(dst, src, off);
        } break;

#if defined(SMART_WITH_QNN)
        case OpType::QNN_FORWARD: {
            auto x     = op->prev[0]->tensor();
            auto out   = op->output();
            auto pos   = op->get_params<QNNForwardParams>().pos;
            auto &mask = op->get_params<QNNForwardParams>().mask;
            m_platform.qnn_backend->forward(m_graph.m_model_id, out, x, pos, mask);
        } break;
        case OpType::QNN_FORWARD_VL: {
            auto x                  = op->prev[0]->tensor();
            auto out                = op->output();
            auto pos                = op->get_params<QNNForwardVLParams>().pos;
            auto &mask              = op->get_params<QNNForwardVLParams>().mask;
            auto &pixel_values_list = op->get_params<QNNForwardVLParams>().pixel_values_list;
            auto &img_infos         = op->get_params<QNNForwardVLParams>().img_infos;
            m_platform.qnn_backend->forward(m_graph.m_model_id, out, x, pixel_values_list, img_infos, pos, mask);
            pixel_values_list.clear();
            img_infos.clear();
        } break;
#endif

        case OpType::QUEST_ATTN: {
            auto q                                 = op->prev[0]->tensor();
            auto out                               = op->output();
            auto [pos, layer_id, regions, n_heads] = op->get_params<QuestAttnParams>();
            m_platform.ggml_backend->quest_attention(out, q, pos, layer_id, regions, n_heads);
        } break;

        case OpType::COS_SIM: {
            auto src0 = op->prev[0]->tensor();
            auto src1 = op->prev[1]->tensor();
            m_platform.ggml_backend->cos_sim(src0, src1);
        } break;

        case OpType::PRINT: {
            auto x    = op->prev[0]->tensor();
            auto size = op->get_params<PrintParams>().size;
            m_platform.ggml_backend->print(x, size);

        } break;

        case OpType::ADD_CACHE: {
            auto src                     = op->prev[0]->tensor();
            auto [L, pos, head_id, is_k] = op->get_params<AddCacheParams>();
            m_platform.ggml_backend->add_cache(src, L, pos, head_id, is_k);
        } break;
        default:
            SMART_ABORT("Unknown OpType: {}", static_cast<int>(op->op));
        }
    }
}
} // namespace smart
