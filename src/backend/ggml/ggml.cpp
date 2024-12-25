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

void GGMLBackend::get_embedding(const Tensor *dst, const Tensor *weight, const std::vector<int> &tokens) const {
    auto embd_tb = static_cast<char *>(weight->get<Buffer>().m_data);
    auto dst_tb  = static_cast<float *>(dst->get<Buffer>().m_data);

    auto dim        = dst->m_shape[0];
    auto batch_size = tokens.size();
    SMART_ASSERT(batch_size == dst->m_shape[1]);
    auto weight_strip = weight->get<Buffer>().m_stride;

    for (size_t i = 0; i < batch_size; i++) {
        auto token = tokens[i];
        auto src   = embd_tb + weight_strip[1] * token;
        SMART_ASSERT(src < embd_tb + weight_strip[2]);
        switch (weight->m_dtype) {
        case DataType::FP32: {
            memcpy(dst_tb + i * dim, src, dim * sizeof(float));
        } break;

        case DataType::GGML_Q4_0: {
            dequantize_row_q4_0((block_q4_0 *)src, dst_tb + i * dim, dim);
        } break;

        case DataType::GGML_Q8_0: {
            dequantize_row_q8_0((block_q8_0 *)src, dst_tb + i * dim, dim);
        } break;

        default:
            SMART_ABORT("unsupported data type: {}", static_cast<int>(weight->m_dtype));
        }
    }
}

void GGMLBackend::matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    auto dst_tensor  = convert_to_ggml(dst);
    auto src0_tensor = convert_to_ggml(src0);
    auto src1_tensor = convert_to_ggml(src1);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        params.thread_pool = (void *)m_thread_pool.get();
        params.barrier_fn  = [](void *opaque) {
            auto thread_pool = (ThreadPool *)opaque;
            // fmt::print("thread waiting for barrier\n");
            thread_pool->barrier();
        };
        params.current_chunk = (atomic_int *)&m_current_chunk;

        smart_compute_forward_mul_mat(&params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get());
    });
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

void GGMLBackend::rmsnorm(const Tensor *out, const Tensor *x, const Tensor *weight, float eps) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);
    auto src1_tensor = convert_to_ggml(weight);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        smart_compute_forward_rms_norm(&params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get(), eps);
    });
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

void GGMLBackend::softmax(const Tensor *out, const Tensor *x) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);

    // fmt::print("Running matmul with {} threads\n", m_thread_pool->size());
    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        smart_compute_forward_soft_max(&params, dst_tensor.get(), src0_tensor.get());
    });
}

void GGMLBackend::rope(
    Tensor *out, const Tensor *src, const std::vector<int> &pos, const ModelConfig::LLMConfig::RopeConfig &rope_cfg
) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(src);
    auto src1_tensor = std::make_unique<ggml_tensor>();
    {
        src1_tensor->data  = (void *)pos.data();
        src1_tensor->type  = GGML_TYPE_I32;
        src1_tensor->ne[0] = pos.size();
        src1_tensor->ne[1] = src1_tensor->ne[2] = src1_tensor->ne[3] = 1;
        src1_tensor->nb[0]                                           = sizeof(int32_t);
        src1_tensor->nb[1] = src1_tensor->nb[2] = src1_tensor->nb[3] = pos.size() * sizeof(int32_t);
    }

    rope_compute_params rope_params = {
        .n_dims      = rope_cfg.n_dims,
        .n_ctx_orig  = rope_cfg.n_ctx_orig,
        .freq_base   = rope_cfg.freq_base,
        .freq_scale  = rope_cfg.freq_scale,
        .ext_factor  = rope_cfg.ext_factor,
        .attn_factor = rope_cfg.attn_factor,
        .beta_fast   = rope_cfg.beta_fast,
        .beta_slow   = rope_cfg.beta_slow,
        .mode        = rope_cfg.rope_type,
    };

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        smart_compute_forward_rope(
            &params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get(), nullptr, &rope_params
        );
    });
}

void GGMLBackend::reset_kv_batch_size(const size_t batch_size) const {
    m_kv->reset_batch_size(batch_size);
}

void GGMLBackend::multihead_attention(
    const Tensor *out, const Tensor *q, const std::vector<int> &pos, const int64_t L, const uint32_t n_heads
) const {
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

void GGMLBackend::add(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    auto dst_tensor  = convert_to_ggml(dst);
    auto src0_tensor = convert_to_ggml(src0);
    auto src1_tensor = convert_to_ggml(src1);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        smart_compute_forward_add(&params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get());
    });
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

void GGMLBackend::copy(const Tensor *dst, const Tensor *src, const int64_t off) const {
    auto dst_ptr = static_cast<float *>(dst->get<Buffer>().m_data) + off;
    auto src_ptr = static_cast<float *>(src->get<Buffer>().m_data);
    memcpy((void *)dst_ptr, src_ptr, src->n_elements() * sizeof(float));
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
    printf("\n{%ld, %ld, %ld, %ld}\n", x->m_shape[3], x->m_shape[2], x->m_shape[1], x->m_shape[0]);
    auto shape  = x->m_shape;
    auto stride = x->get<Buffer>().m_stride;
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

bool GGMLBackend::is_contiguous(const Tensor *tensor, int n) const {
    SMART_ASSERT(n >= 0 && n <= 2);
    if (n == 0) {
        return ggml_is_contiguous_0(convert_to_ggml(tensor).get());
    } else if (n == 1) {
        return ggml_is_contiguous_1(convert_to_ggml(tensor).get());
    } else if (n == 2) {
        return ggml_is_contiguous_2(convert_to_ggml(tensor).get());
    }
    return false;
}

void GGMLBackend::add_cache(const Tensor *src, size_t L, const std::vector<int> &pos, size_t head_id, bool is_k) {
    SMART_UNUSED(head_id);
    auto kv_dim       = m_kv->m_kv_dim;
    float *dst_base   = nullptr;
    float *src_buffer = static_cast<float *>(src->get<ggml::Buffer>().m_data); // (kv_dim, batch_size, 1, 1)
    if (is_k) {
        dst_base = m_kv->chunk.key_buffer[L].data(); // [seq_len * kv_dim]
    } else {
        dst_base = m_kv->chunk.value_buffer[L].data();
    }

    for (auto i = 0; auto p : pos) {
        // dst[L][p] <- src[i]
        memcpy(dst_base + p * kv_dim, src_buffer + i * kv_dim, kv_dim * sizeof(float));
        i += 1;
    }
}

} // namespace smart::ggml
