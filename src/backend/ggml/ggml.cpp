#include "ggml.hpp"

#include "backend/ggml/buffer.hpp"
#include "ggml.h"
#include "model/module/region.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace smart::ggml {

void GGMLBackend::matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    auto dst_tensor  = convert_to_ggml(dst);
    auto src0_tensor = convert_to_ggml(src0);
    auto src1_tensor = convert_to_ggml(src1);

    // fmt::print("Running matmul with {} threads\n", m_thread_pool->size());
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
    // fmt::print("Finished matmul with {} threads\n", m_thread_pool->size());
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

// TODO: Rope's pos should be a tensor and we need rope_base (llama2 = 10000, llama3 = 300000 ...)
void GGMLBackend::rope(Tensor *q_out, Tensor *k_out, const Tensor *q, const Tensor *k, const Tensor *pos) const {
    auto dim                = q->m_shape[0];
    auto head_size          = dim / m_config->tf_cfg.n_heads;
    auto kv_dim             = (m_config->tf_cfg.dim * m_config->tf_cfg.n_kv_heads) / m_config->tf_cfg.n_heads;
    const int32_t *pos_data = static_cast<int32_t *>(pos->get<Buffer>().m_data);

    memcpy(q_out->get<Buffer>().m_data, q->get<Buffer>().m_data, q->n_elements() * sizeof(float));
    memcpy(k_out->get<Buffer>().m_data, k->get<Buffer>().m_data, k->n_elements() * sizeof(float));

    for (size_t i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        float freq   = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val    = pos_data[0] * freq;
        float fcr    = cosf(val);
        float fci    = sinf(val);
        int rotn     = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        for (int v = 0; v < rotn; v++) {
            float *vec = v == 0 ? static_cast<float *>(q_out->get<Buffer>().m_data)
                                : static_cast<float *>(k_out->get<Buffer>().m_data);
            float v0   = vec[i];
            float v1   = vec[i + 1];
            vec[i]     = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

void GGMLBackend::multihead_attention(
    const Tensor *out,
    const Tensor *q,
    const Tensor *key_cache,
    const Tensor *val_cache,
    const Tensor *pos,
    const int64_t L
) const {
    auto dim                = m_config->tf_cfg.dim;
    auto kv_dim             = (m_config->tf_cfg.dim * m_config->tf_cfg.n_kv_heads) / m_config->tf_cfg.n_heads;
    auto kv_mul             = m_config->tf_cfg.n_heads / m_config->tf_cfg.n_kv_heads;
    auto head_size          = dim / m_config->tf_cfg.n_heads;
    uint64_t loff           = L * m_config->tf_cfg.seq_len * kv_dim;
    const int32_t *pos_data = static_cast<int32_t *>(pos->get<Buffer>().m_data);
    auto att                = std::vector<float>(m_config->tf_cfg.n_heads * m_config->tf_cfg.seq_len);

    uint32_t h = 0;
    for (h = 0; h < m_config->tf_cfg.n_heads; h++) {
        auto q_buf   = static_cast<float *>(q->get<Buffer>().m_data) + h * head_size;
        auto att_buf = att.data() + h * m_config->tf_cfg.seq_len;

        for (auto t = 0; t <= pos_data[0]; t++) {
            auto k =
                static_cast<float *>(key_cache->get<Buffer>().m_data) + loff + t * kv_dim + (h / kv_mul) * head_size;
            auto score = 0.0f;

            for (size_t i = 0; i < head_size; i++) {
                score += q_buf[i] * k[i];
            }

            score /= sqrtf(head_size);
            att_buf[t] = score;
        }

        softmax_internal(att_buf, att_buf, pos_data[0] + 1);

        auto xb = static_cast<float *>(out->get<Buffer>().m_data) + h * head_size;
        memset(xb, 0, head_size * sizeof(float));

        for (auto t = 0; t <= pos_data[0]; t++) {
            auto v =
                static_cast<float *>(val_cache->get<Buffer>().m_data) + loff + t * kv_dim + (h / kv_mul) * head_size;
            auto a = att_buf[t];

            for (size_t i = 0; i < head_size; i++) {
                xb[i] += a * v[i];
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
    for (size_t i = 0; i < m_config->tf_cfg.hidden_dim; i++) {
        float val = static_cast<float *>(hb->get<Buffer>().m_data)[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= static_cast<float *>(hb2->get<Buffer>().m_data)[i];
        static_cast<float *>(out->get<Buffer>().m_data)[i] = val;
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
    std::vector<Region> &regions
) const {
    auto dim         = m_config->tf_cfg.dim;
    auto kv_dim      = (m_config->tf_cfg.dim * m_config->tf_cfg.n_kv_heads) / m_config->tf_cfg.n_heads;
    auto kv_mul      = m_config->tf_cfg.n_heads / m_config->tf_cfg.n_kv_heads;
    auto head_size   = dim / m_config->tf_cfg.n_heads;
    uint64_t loff    = L * m_config->tf_cfg.seq_len * kv_dim;
    auto pos_data    = static_cast<int32_t *>(pos->get<Buffer>().m_data);
    auto att         = std::vector<float>(m_config->tf_cfg.n_heads * m_config->tf_cfg.seq_len, -INFINITY);
    uint32_t n_init  = 1;
    uint32_t topK    = 2; // head + topK + tails // cosine similarity
    uint32_t n_local = 4;

    // Update regions
    if (regions[regions.size() - 1].is_full()) {
        regions.emplace_back(kv_dim, REGION_TOKENS);
    }
    regions[regions.size() - 1].update_score(
        static_cast<float *>(key_cache->get<Buffer>().m_data), m_config->tf_cfg.seq_len, L, pos_data[0]
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
    for (h = 0; h < m_config->tf_cfg.n_heads; h++) {
        auto q_   = static_cast<float *>(q->get<Buffer>().m_data) + h * head_size;
        auto att_ = att.data() + h * m_config->tf_cfg.seq_len;

        for (auto &r : top_regions) {
            for (auto &t : r.region_tensors) {
                uint64_t off = t.L * m_config->tf_cfg.seq_len * kv_dim + t.pos * kv_dim + (h / kv_mul) * head_size;
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

} // namespace smart::ggml
