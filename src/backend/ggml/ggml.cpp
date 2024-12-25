#include "ggml.hpp"

#include "backend/ggml/buffer.hpp"
#include "core/data_type.hpp"
#include "ggml-quants.h"
#include "ggml.h"
#include "model/module/region.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace smart::ggml {

void GGMLBackend::plan(std::vector<std::shared_ptr<OpNode>> &ops) {
    size_t max_work_size = 0;
    for (auto op : ops) {
        size_t cur = 0;

        const int n_tasks = get_n_tasks(op);

        switch (op->op) {
        // custom ops
        case OpType::COS_SIM:
        case OpType::MHA:
        case OpType::QUEST_ATTN:
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
            auto x      = op->prev[0]->tensor();
            auto weight = op->prev[1]->tensor();

            const enum ggml_type vec_dot_type = get_vec_dot_type(x);
            if (ggml::convert_datatype_to_ggml(weight->m_dtype) != vec_dot_type) {
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

#if defined(SMART_WITH_QNN)
        case OpType::QNN_FORWARD: {
        } break;
#endif

        default:
            SMART_ABORT("unsupported op type: {}", static_cast<int>(op->op));
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

void GGMLBackend::rmsnorm_internal(float *o, float *x, float *weight, int64_t size) const {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void GGMLBackend::softmax_internal(float *out, float *x, size_t size) const {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (size_t i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }
    // normalize
    for (size_t i = 0; i < size; i++) {
        out[i] /= sum;
    }
}

void GGMLBackend::reset_kv_batch_size(const size_t batch_size) const {
    m_kv->reset_batch_size(batch_size);
}

void GGMLBackend::multihead_attention(
    const Tensor *out, const Tensor *q, const std::vector<int> &pos, const int64_t L, const uint32_t n_heads
) const {
    fmt::println("This function is deprecated!");

    auto dim        = q->m_shape[0];
    auto kv_dim     = m_kv->m_kv_dim;
    auto seq_len    = m_kv->m_n_ctx;
    auto kv_mul     = dim / kv_dim;
    auto head_size  = dim / n_heads;
    auto batch_size = pos.size();
    std::vector<std::vector<float>> att(batch_size);

    for (size_t p = 0; p < batch_size; p++) { // batch size
        auto out_buf = static_cast<float *>(out->get<Buffer>().m_data) + p * out->m_shape[0];
        att[p]       = std::vector<float>(n_heads * seq_len, 0);
        // auto mask_buf = static_cast<float *>(mask->get<Buffer>().m_data);
        // SMART_UNUSED(mask_buf);
        uint32_t h = 0;
        for (h = 0; h < n_heads; h++) {
            auto q_buf   = static_cast<float *>(q->get<Buffer>().m_data) + p * q->m_shape[0] + h * head_size;
            auto att_buf = att[p].data() + h * seq_len;

            for (auto t = 0; t <= pos[p]; t++) {
                auto k     = m_kv->chunk.key_buffer[L].data() + t * kv_dim + (h / kv_mul) * head_size;
                auto score = 0.0f;

                for (size_t i = 0; i < head_size; i++) {
                    score += q_buf[i] * k[i];
                }

                score /= sqrtf(head_size);
                att_buf[t] = score;
                // att_buf[t] += mask_buf[p * mask->m_shape[1] * mask->m_shape[2] + t];
            }

            softmax_internal(att_buf, att_buf, pos[p] + 1);

            auto xb = out_buf + h * head_size;
            memset(xb, 0, head_size * sizeof(float));

            for (auto t = 0; t <= pos[p]; t++) {
                auto v = m_kv->chunk.value_buffer[L].data() + t * kv_dim + (h / kv_mul) * head_size;
                auto a = att_buf[t];

                for (size_t i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
    }
}

void GGMLBackend::silu_hadamard(const Tensor *out, const Tensor *hb, const Tensor *hb2) const {
    SMART_ASSERT(is_contiguous(out, 0));
    SMART_ASSERT(is_contiguous(hb, 0));
    SMART_ASSERT(is_contiguous(hb2, 0));
    float *out_data = static_cast<float *>(out->get<Buffer>().m_data);
    float *hb_data  = static_cast<float *>(hb->get<Buffer>().m_data);
    float *hb2_data = static_cast<float *>(hb2->get<Buffer>().m_data);

    for (size_t j = 0; j < hb->n_elements(); j++) {
        float val = hb_data[j];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= hb2_data[j];
        out_data[j] = val;
    }
}

void GGMLBackend::quest_attention(
    const Tensor *out,
    const Tensor *q,
    const std::vector<int> &pos,
    const int64_t L,
    std::vector<Region> &regions,
    const uint32_t n_heads
) const { // No batch size -- no need
    auto dim        = q->m_shape[0];
    auto kv_dim     = m_kv->m_kv_dim;
    auto seq_len    = m_kv->m_n_ctx;
    auto kv_mul     = dim / kv_dim;
    auto head_size  = dim / n_heads;
    auto batch_size = pos.size();

    auto att         = std::vector<float>(n_heads * seq_len, -INFINITY);
    uint32_t n_init  = 1;
    uint32_t topK    = 2; // head + topK + tails // cosine similarity
    uint32_t n_local = 4;

    // Update regions TODO: extract this step as a single step
    for (size_t p = 0; p < batch_size; p++) {
        if (regions[regions.size() - 1].is_full()) {
            regions.emplace_back(kv_dim, REGION_TOKENS);
        }
        regions[regions.size() - 1].update_score(m_kv->chunk.key_buffer[L].data(), seq_len, L, pos[p]);
    }
    if (pos.size() > 1) { // prefill
        multihead_attention(out, q, pos, L, n_heads);
        return;
    }

    // Multihead attention for target regions' tokens
    // Calculate scores
    auto n_regions = regions.size(); // all tokens = regions * (REGION_TOKENS - 1) + regions.back().n_tokens

    for (uint32_t i = 0; i < n_regions; i++) {
        regions[i].score(static_cast<float *>(q->get<Buffer>().m_data), kv_mul);
    }
    // Top-k regions
    std::vector<Region> top_regions(regions);
    if (n_regions > topK + n_init + n_local) {
        std::sort(top_regions.begin() + n_init, top_regions.end() - n_local, [](const Region &a, const Region &b) {
            return a.final_score > b.final_score;
        });
        top_regions.resize(std::min((long)topK, (long)n_regions));
        for (uint32_t i = 0; i < n_init; i++) {
            top_regions.push_back(regions[i]);
        }
        for (uint32_t i = 0; i < n_local; i++) {
            top_regions.push_back(regions[n_regions - 1 - i]);
        }
    }

    uint32_t h = 0;
    for (h = 0; h < n_heads; h++) {
        auto q_   = static_cast<float *>(q->get<Buffer>().m_data) + h * head_size;
        auto att_ = att.data() + h * seq_len;

        for (auto &r : top_regions) {
            for (auto &t : r.region_tensors) {
                auto k = m_kv->chunk.key_buffer[L].data() + t.pos * kv_dim + (h / kv_mul) * head_size;

                auto score = 0.0f;
                for (uint32_t i = 0; i < head_size; i++) {
                    score += q_[i] * k[i];
                }

                score /= sqrtf(head_size);
                att_[t.pos] = score;
            }
        }

        softmax_internal(att_, att_, pos[0] + 1);

        auto xb_ = static_cast<float *>(out->get<Buffer>().m_data) + h * head_size;
        memset(xb_, 0, head_size * sizeof(float));

        for (auto t = 0; t <= pos[0]; t++) {
            auto v = m_kv->chunk.value_buffer[L].data() + t * kv_dim + (h / kv_mul) * head_size;
            auto a = att_[t];

            for (uint32_t i = 0; i < head_size; i++) {
                xb_[i] += a * v[i];
            }
        }
    }
}

void GGMLBackend::cos_sim(const Tensor *src0, const Tensor *src1) const {
    cos_sim_internal(
        static_cast<float *>(src0->get<Buffer>().m_data),
        static_cast<float *>(src1->get<Buffer>().m_data),
        src0->n_elements()
    );
}

void GGMLBackend::cos_sim_internal(float *x_, float *y_, size_t size) const {
    float dot_product = 0.0f;
    float magnitude_x = 0.0f;
    float magnitude_y = 0.0f;

    for (size_t i = 0; i < size; i++) {
        dot_product += x_[i] * y_[i];
        magnitude_x += x_[i] * x_[i];
        magnitude_y += y_[i] * y_[i];
    }

    magnitude_x = std::sqrt(magnitude_x);
    magnitude_y = std::sqrt(magnitude_y);

    auto out_ = dot_product / (magnitude_x * magnitude_y);
    if (out_ < 0.99) {
        fmt::print(stderr, "\ncos_sim: {}\n", out_);
    }
}

void GGMLBackend::print(const Tensor *x, size_t size) const {
    SMART_UNUSED(size);
    SMART_ASSERT(x->m_dtype == DataType::FP32);
    auto shape  = x->m_shape;
    auto stride = x->get<Buffer>().m_stride;
    printf("\n{%ld, %ld, %ld, %ld}\n", shape[3], shape[2], shape[1], shape[0]);
    printf("\n{%ld, %ld, %ld, %ld}\n", stride[3], stride[2], stride[1], stride[0]);
    for (size_t i3 = 0; i3 < shape[3]; i3++) {
        for (size_t i2 = 0; i2 < shape[2]; i2++) {
            for (size_t i1 = 0; i1 < shape[1]; i1++) {
                for (size_t i0 = 0; i0 < shape[0]; i0++) {
                    float *ptr = (float *)((char *)x->get<Buffer>().m_data + i3 * stride[3] + i2 * stride[2] +
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
    SMART_UNUSED(head_id);

    auto kv_dim       = m_kv->m_kv_dim;
    auto batch_size   = pos.size();
    auto cur_position = m_kv->kv_cache->position;
    SMART_ASSERT(batch_size == m_kv->m_batch_size);

    float *src_k  = static_cast<float *>(k->get<ggml::Buffer>().m_data); // (kv_dim, batch_size, 1, 1)
    float *src_v  = static_cast<float *>(v->get<ggml::Buffer>().m_data); // (kv_dim, batch_size, 1, 1)
    float *dst_kb = m_kv->chunk.key_buffer[L].data() + kv_dim * cur_position;
    float *dst_vb = m_kv->chunk.value_buffer[L].data() + kv_dim * cur_position;
    memcpy(dst_kb, src_k, kv_dim * batch_size * sizeof(float));
    memcpy(dst_vb, src_v, kv_dim * batch_size * sizeof(float));
}

void GGMLBackend::transpose(const Tensor *out, const Tensor *x) const {
    Buffer::Stride stride{x->get<Buffer>().m_stride};
    stride[0] = x->get<Buffer>().m_stride[1];
    stride[1] = x->get<Buffer>().m_stride[0];

    out->get<Buffer>().m_data   = x->get<Buffer>().m_data;
    out->get<Buffer>().m_stride = stride;
}

} // namespace smart::ggml
