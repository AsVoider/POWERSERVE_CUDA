# include "ggml.hpp"

# include "backend/ggml/buffer.hpp"
#include "ggml.h"
#include "model/module/region.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace smart::ggml {

void  GGMLBackend::matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
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

void GGMLBackend::rmsnorm(const Tensor *out, const Tensor *x, const Tensor *weight) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);
    auto src1_tensor = convert_to_ggml(weight);

    // fmt::print("Running matmul with {} threads\n", m_thread_pool->size());
    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        smart_compute_forward_rms_norm(&params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get());
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

void GGMLBackend::rope(Tensor *out, const Tensor *src, const Tensor *pos, const RopeConfig &rope_cfg) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(src);
    auto src1_tensor = convert_to_ggml(pos);

    rope_compute_params rope_params = {
        .n_dims      = rope_cfg.n_dims,
        .n_ctx_orig  = rope_cfg.n_ctx_orig,
        .freq_base   = rope_cfg.freq_base,
        .freq_scale  = rope_cfg.freq_scale,
        .ext_factor  = rope_cfg.ext_factor,
        .attn_factor = rope_cfg.attn_factor,
        .beta_fast   = rope_cfg.beta_fast,
        .beta_slow   = rope_cfg.beta_slow
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

void GGMLBackend::multihead_attention(
    const Tensor *out,
    const Tensor *q,
    const Tensor *key_cache,
    const Tensor *val_cache,
    const Tensor *pos,
    const int64_t L,
    const uint32_t n_heads
) const {
    auto dim                = q->m_shape[0];
    auto kv_dim             = key_cache->m_shape[0];
    auto seq_len            = key_cache->m_shape[1];
    auto kv_mul             = dim / kv_dim;
    auto head_size          = dim / n_heads;
    uint64_t loff           = L * seq_len * kv_dim;
    const int32_t *pos_data = static_cast<int32_t *>(pos->get<Buffer>().m_data);
    std::vector<std::vector<float>> att(pos->m_shape[0]);

    for (size_t p = 0; p < pos->m_shape[0]; p++) { // batch size
        auto out_buf = static_cast<float *>(out->get<Buffer>().m_data) + p * out->m_shape[0];
        att[p]       = std::vector<float>(n_heads * seq_len, 0);
        // auto mask_buf = static_cast<float *>(mask->get<Buffer>().m_data);
        // SMART_UNUSED(mask_buf);
        uint32_t h = 0;
        for (h = 0; h < n_heads; h++) {
            auto q_buf   = static_cast<float *>(q->get<Buffer>().m_data) + p * q->m_shape[0] + h * head_size;
            auto att_buf = att[p].data() + h * seq_len;

            for (auto t = 0; t <= pos_data[p]; t++) {
                auto k = static_cast<float *>(key_cache->get<Buffer>().m_data) + loff + t * kv_dim +
                         (h / kv_mul) * head_size;
                auto score = 0.0f;

                for (size_t i = 0; i < head_size; i++) {
                    score += q_buf[i] * k[i];
                }

                score /= sqrtf(head_size);
                att_buf[t] = score;
                // att_buf[t] += mask_buf[p * mask->m_shape[1] * mask->m_shape[2] + t];
            }

            softmax_internal(att_buf, att_buf, pos_data[p] + 1);

            auto xb = out_buf + h * head_size;
            memset(xb, 0, head_size * sizeof(float));

            for (auto t = 0; t <= pos_data[p]; t++) {
                auto v = static_cast<float *>(val_cache->get<Buffer>().m_data) + loff + t * kv_dim +
                         (h / kv_mul) * head_size;
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

    // fmt::print("Running matmul with {} threads\n", m_thread_pool->size());
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
    const Tensor *key_cache,
    const Tensor *val_cache,
    const Tensor *pos,
    const int64_t L,
    std::vector<Region> &regions,
    const uint32_t n_heads
) const { // No batch size -- no need
    auto dim       = q->m_shape[0];
    auto kv_dim    = key_cache->m_shape[0];
    auto seq_len   = key_cache->m_shape[1];
    auto kv_mul    = dim / kv_dim;
    auto head_size = dim / n_heads;
    uint64_t loff  = L * seq_len * kv_dim;

    auto pos_data    = static_cast<int32_t *>(pos->get<Buffer>().m_data);
    auto att         = std::vector<float>(n_heads * seq_len, -INFINITY);
    uint32_t n_init  = 1;
    uint32_t topK    = 2; // head + topK + tails // cosine similarity
    uint32_t n_local = 4;

    // Update regions
    if (regions[regions.size() - 1].is_full()) {
        regions.emplace_back(kv_dim, REGION_TOKENS);
    }
    regions[regions.size() - 1].update_score(
        static_cast<float *>(key_cache->get<Buffer>().m_data), seq_len, L, pos_data[0]
    );

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
                uint64_t off = t.L * seq_len * kv_dim + t.pos * kv_dim + (h / kv_mul) * head_size;
                auto k       = static_cast<float *>(key_cache->get<Buffer>().m_data) + off;

                auto score = 0.0f;
                for (uint32_t i = 0; i < head_size; i++) {
                    score += q_[i] * k[i];
                }

                score /= sqrtf(head_size);
                att_[t.pos] = score;
            }
        }

        softmax_internal(att_, att_, pos_data[0] + 1);

        auto xb_ = static_cast<float *>(out->get<Buffer>().m_data) + h * head_size;
        memset(xb_, 0, head_size * sizeof(float));

        for (auto t = 0; t <= pos_data[0]; t++) {
            auto v =
                static_cast<float *>(val_cache->get<Buffer>().m_data) + loff + t * kv_dim + (h / kv_mul) * head_size;
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
} // namespace smart::ggml
