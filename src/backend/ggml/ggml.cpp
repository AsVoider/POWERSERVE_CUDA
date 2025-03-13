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

#include "ggml.hpp"

#include "core/data_type.hpp"
#include "cpu_buffer.hpp"
#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace powerserve::ggml {

void GGMLBackend::plan(std::vector<std::shared_ptr<OpNode>> &ops) {
    size_t max_work_size = 0;
    for (auto op : ops) {
        if (op->compute_backend != TensorBackend::GGML_CPU) {
            continue;
        }
        size_t cur = 0;

        const int n_tasks = get_n_tasks(op);

        switch (op->op) {
        // custom ops
        case OpType::SILU_HADAMARD:
        case OpType::ADD_CACHE:
        case OpType::TRANSPOSE:
        case OpType::PRINT:
        case OpType::VIEW:
        case OpType::COPY: {
        } break;

        case OpType::PERMUTE:
        case OpType::CONT:
        case OpType::GET_MASK:
        case OpType::GET_EMBEDDING: {
            max_work_size = 0;
        } break;

        case OpType::ADD: {
            auto a = op->prev[0]->tensor();
            if (a->is_quantized()) {
                cur = ggml_type_size(GGML_TYPE_F32) * a->m_shape[0] * n_tasks;
            }
        } break;

        case OpType::MAT_MUL: {
            auto weight = op->prev[0]->tensor();
            auto x      = op->prev[1]->tensor();
            // printf("x shape is %ld, %ld, %ld, %ld, we shape is %ld, %ld, %ld, %ld\n",
            //     x->m_shape[0], x->m_shape[1], x->m_shape[2], x->m_shape[3], weight->m_shape[0], weight->m_shape[1],
            //     weight->m_shape[2], weight->m_shape[3]);
            const enum ggml_type vec_dot_type = get_vec_dot_type(x);
            if (convert_datatype_to_ggml(weight->m_dtype) != vec_dot_type) {
                cur = ggml_row_size(vec_dot_type, weight->n_elements());
            }
        } break;

        case OpType::SOFTMAX_EXT:
        case OpType::SOFTMAX:
        case OpType::ROPE: {
            auto dst = op->next[0]->tensor();
            cur      = ggml_type_size(GGML_TYPE_F32) * dst->m_shape[0] * n_tasks;
        } break;

        case OpType::RMS_NORM: {
        } break;

#if defined(POWERSERVE_WITH_QNN)
        case OpType::QNN_FORWARD: {
        } break;
        case OpType::QNN_FORWARD_VL: {
        } break;
#endif

        default:
            POWERSERVE_ABORT("unsupported op type: {}", static_cast<int>(op->op));
        }

        max_work_size = std::max(max_work_size, cur);
    }

    setup_work_data(max_work_size);
}

void GGMLBackend::setup_work_data(size_t work_size) {
    if (work_size <= m_wdata.size()) {
        return;
    }
    if (work_size > 0) {
        work_size += get_cache_line_size() * num_threads;
    }

    m_wdata.resize(work_size);
    m_params.wdata = m_wdata.data();
    m_params.wsize = m_wdata.size();
}

void GGMLBackend::advance(const size_t &size) {
    m_kv->advanced_kv_cache_size(size);
}

void GGMLBackend::reset_kv_batch_size(const size_t &batch_size) {
    m_kv->reset_kv_batch_size(batch_size);
}

void GGMLBackend::silu_hadamard(const Tensor *out, const Tensor *hb, const Tensor *hb2) const {
    POWERSERVE_ASSERT(is_contiguous(out, 0));
    POWERSERVE_ASSERT(is_contiguous(hb, 0));
    POWERSERVE_ASSERT(is_contiguous(hb2, 0));
    float *out_data = static_cast<float *>(out->m_data->m_data_host);
    float *hb_data  = static_cast<float *>(hb->m_data->m_data_host);
    float *hb2_data = static_cast<float *>(hb2->m_data->m_data_host);

    for (size_t j = 0; j < hb->n_elements(); j++) {
        float val = hb_data[j];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= hb2_data[j];
        out_data[j] = val;
    }
}

void GGMLBackend::print(const Tensor *x, size_t size) const {
    POWERSERVE_UNUSED(size);
    POWERSERVE_ASSERT(x->m_dtype == DataType::FP32);
    auto shape  = x->m_shape;
    auto stride = x->m_data->m_stride;
    printf("\n{%ld, %ld, %ld, %ld}\n", shape[3], shape[2], shape[1], shape[0]);
    printf("\n{%ld, %ld, %ld, %ld}\n", stride[3], stride[2], stride[1], stride[0]);
    for (size_t i3 = 0; i3 < shape[3]; i3++) {
        for (size_t i2 = 0; i2 < shape[2]; i2++) {
            for (size_t i1 = 0; i1 < shape[1]; i1++) {
                for (size_t i0 = 0; i0 < shape[0]; i0++) {
                    float *ptr = (float *)((char *)x->m_data->m_data_host + i3 * stride[3] + i2 * stride[2] +
                                           i1 * stride[1] + i0 * stride[0]);
                    // printf("[%ld][%ld][%ld][%ld] = %.6f\n", i3, i2, i1, i0, (double)*ptr);
                    printf("%.6f\n", (double)*ptr);
                }
            }
        }
    }
    exit(0);
}

void GGMLBackend::add_cache(const Tensor *k, const Tensor *v, size_t L, const std::vector<int> &pos, size_t head_id) {
    fmt::println("This function is deprecated!");
    POWERSERVE_UNUSED(head_id);

    auto kv_dim       = m_kv->kv_shape.kv_dim;
    auto batch_size   = pos.size();
    auto cur_position = m_kv->kv_shape.kv_size;
    POWERSERVE_ASSERT(batch_size == m_kv->kv_shape.batch_size);

    float *src_k  = static_cast<float *>(k->m_data->m_data_host); // (kv_dim, batch_size, 1, 1)
    float *src_v  = static_cast<float *>(v->m_data->m_data_host); // (kv_dim, batch_size, 1, 1)
    float *dst_kb = reinterpret_cast<float *>(m_kv->k_cache[L].cache_data_ptr + kv_dim * cur_position * sizeof(float)); // fixed to use k_cache for destination
    float *dst_vb = reinterpret_cast<float *>(m_kv->v_cache[L].cache_data_ptr + kv_dim * cur_position * sizeof(float)); // fixed to use v_cache for destination
    memcpy(dst_kb, src_k, kv_dim * batch_size * sizeof(float));
    memcpy(dst_vb, src_v, kv_dim * batch_size * sizeof(float));
}

void GGMLBackend::transpose(const Tensor *out, const Tensor *x) const {
    Stride stride{x->m_data->m_stride};
    stride[0] = x->m_data->m_stride[1];
    stride[1] = x->m_data->m_stride[0];

    out->m_data->m_data_host   = x->m_data->m_data_host;
    out->m_data->m_stride = stride;
}

void GGMLBackend::setup_threadpool() {
    m_thread_pool = std::make_unique<ThreadPool>(m_thread_config);
}

void GGMLBackend::reset_threadpool() {
    POWERSERVE_LOG_DEBUG("reset_threadpool");
    m_thread_pool.reset();
}

std::pair<Tensor *, Tensor *> GGMLBackend::get_kv_cache(size_t layer_id) {
    return m_kv->get_cache(layer_id);
}

void GGMLBackend::graph_compute(std::vector<std::shared_ptr<OpNode>> &ops) {
    plan(ops);

    for (auto &op : ops) {
        switch (op->op) {
        case OpType::GET_EMBEDDING: {
            auto weight   = op->prev[0]->tensor();
            auto out      = op->output();
            auto [tokens] = op->get_params<GetEmbeddingParams>();
            get_embedding(out, weight, tokens);
        } break;

        case OpType::ADD: {
            auto a   = op->prev[0]->tensor();
            auto b   = op->prev[1]->tensor();
            auto out = op->output();
            add(out, a, b);
        } break;

        case OpType::MAT_MUL: {
            auto a   = op->prev[0]->tensor();
            auto b   = op->prev[1]->tensor();
            auto out = op->output();
            matmul(out, a, b);
        } break;

        case OpType::RMS_NORM: {
            auto x      = op->prev[0]->tensor();
            auto weight = op->prev[1]->tensor();
            auto out    = op->output();
            auto [eps]  = op->get_params<RMSNormParams>();
            rmsnorm(out, x, weight, eps);
        } break;

        case OpType::SILU_HADAMARD: {
            auto gate = op->prev[0]->tensor();
            auto up   = op->prev[1]->tensor();
            auto out  = op->output();
            silu_hadamard(out, gate, up);
        } break;

        case OpType::ROPE: {
            auto src             = op->prev[0]->tensor();
            auto rope_factors    = op->prev[1]->tensor();
            auto out             = op->next[0]->tensor();
            auto [pos, rope_cfg] = op->get_params<RopeParams>();
            rope(out, src, rope_factors, pos, rope_cfg);
        } break;

        case OpType::SOFTMAX: {
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            softmax(out, x);
        } break;

        case OpType::COPY: {
            auto dst = op->prev[0]->tensor();
            auto src = op->prev[1]->tensor();
            copy(dst, src);
        } break;

        case OpType::PRINT: {
            auto x    = op->prev[0]->tensor();
            auto size = op->get_params<PrintParams>().size;
            print(x, size);
        } break;

        case OpType::ADD_CACHE: {
            auto k                 = op->prev[0]->tensor();
            auto v                 = op->prev[1]->tensor();
            auto [L, pos, head_id] = op->get_params<AddCacheParams>();
            add_cache(k, v, L, pos, head_id);
        } break;

        case OpType::PERMUTE: {
            // auto x      = op->prev[0]->tensor();
            // auto out    = op->output();
            // auto [axes] = op->get_params<PermuteParams>();
            // permute(out, x, axes);
        } break;

        case OpType::CONT: {
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            cont(out, x);
        } break;

        case OpType::VIEW: {
            // auto out                       = op->output();
            // auto [stride, offset]          = op->get_params<ViewParams>();
            // out->m_data->m_stride = stride;
            // out->m_data->m_data   = (char *)out->m_data->m_data + offset;
        } break;

        case OpType::SOFTMAX_EXT: {
            auto out               = op->output();
            auto x                 = op->prev[0]->tensor();
            auto mask              = op->prev[1]->tensor();
            auto [scale, max_bias] = op->get_params<SoftmaxExtParams>();

            softmax_ext(out, x, mask, scale, max_bias);
        } break;

        case OpType::GET_MASK: {
            auto out         = op->output();
            auto [mask, pos] = op->get_params<GetMaskParams>();
            auto n_kv        = out->m_shape[0];
            auto batch_size  = out->m_shape[1];

            POWERSERVE_ASSERT(out->m_dtype == DataType::FP32);
            auto mask_buf = (float *)out->m_data->m_data_host;
            for (size_t i = 0; i < batch_size; i++) {
                size_t cur_pos = pos[i];
                for (size_t j = 0; j < n_kv; j++) {
                    mask_buf[j + i * n_kv] = (j <= cur_pos) ? 0.f : -INFINITY;
                }
            }
        } break;

        case OpType::TRANSPOSE: {
            // auto x   = op->prev[0]->tensor();
            // auto out = op->output();
            // transpose(out, x);
        } break;

        default:
            POWERSERVE_ABORT("Unknown OpType: {}", static_cast<int>(op->op));
        }
    }
}

} // namespace powerserve::ggml
