#include "ggml.hpp"
#include "ggml.h"
#include "graph/node.hpp"
#include <cmath>
#include <cstdint>

namespace smart {
namespace ggml {

void dequantize_row_q8_0(const block_q8_0 *x, float * y, int64_t k) {
    static const int qk = QK8_0;

    SMART_ASSERT(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

void dequantize_row_q4_0(const block_q4_0 * x, float * y, int64_t k) {
    static const int qk = QK4_0;

    SMART_ASSERT(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}
} // namespace ggml

void GGMLBackend::matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    OpTensor dst_ = {
        (void *)dst->data,
        ggml::convert_datatype_to_ggml(dst->dtype),
        {dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]},
        {dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]}
    };
    OpTensor src0_ = {
        (void *)src0->data,
        ggml::convert_datatype_to_ggml(src0->dtype),
        {src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]},
        {src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]}
    };
    OpTensor src1_ = {
        (void *)src1->data,
        ggml::convert_datatype_to_ggml(src1->dtype),
        {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]},
        {src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]}
    };

    ggml_compute_forward_op_mul_mat(&params, &dst_, &src0_, &src1_);
}

void GGMLBackend::rmsnorm_internal(float* o, float* x, float* weight, int64_t size) {
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

void GGMLBackend::rmsnorm(const Tensor *o, const Tensor *x, const Tensor *weight) {
    auto size = x->ne[0];

    rmsnorm_internal((float *)o->data, (float *)x->data, (float *)weight->data, size);
}

void GGMLBackend::softmax_internal(float* x, int64_t size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void GGMLBackend::softmax(const Tensor *x, int64_t size) {
    softmax_internal((float *)x->data, size);
}

void GGMLBackend::rope(const Tensor *q, const Tensor *k, const int64_t pos) {
    auto dim = config->dim;
    auto head_size = dim / config->n_heads;
    auto kv_dim =  (config->dim * config->n_kv_heads) / config->n_heads;

    
    for (int i = 0; i < dim; i+=2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        for (int v = 0; v < rotn; v++) {
            float* vec = v == 0 ? (float *)q->data : (float *)k->data; // the vector to rotate (query or key)
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
}

// TODO: Need to be split
// prev: q, att, key_cache, val_cache, xb, pos, L; next: {att, xb}
void GGMLBackend::multihead_attention(const Tensor *q, const Tensor *att, const Tensor *key_cache, const Tensor *val_cache, const Tensor *xb, const int64_t pos, const int64_t L) {
    auto dim = config->dim;
    auto kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
    auto kv_mul = config->n_heads / config->n_kv_heads;
    auto head_size = dim / config->n_heads;
    uint64_t loff =  L * config->seq_len * kv_dim;
    
    uint32_t h = 0;
    for (h = 0; h < config->n_heads; h++) {
        auto q_   = (float *)q->data + h *head_size;
        auto att_ = (float *)att->data + h * config->seq_len;
        
        for (auto t = 0; t <= pos; t++) {
            auto k = (float *)key_cache->data + loff + t * kv_dim + (h / kv_mul) * head_size;
            auto score = 0.0f;
            
            for (auto i = 0; i < head_size; i++) {
                score += q_[i] * k[i];
            }
        
            score /= sqrtf(head_size);
            att_[t] = score;
        }

        softmax_internal(att_, pos + 1);

        auto xb_ = (float *)xb->data + h * head_size;
        memset(xb_, 0, head_size * sizeof(float));

        for (auto t = 0; t <= pos; t++) {
            auto v = (float *)val_cache->data + loff + t * kv_dim + (h / kv_mul) * head_size;
            auto a = att_[t];

            for (auto i = 0; i < head_size; i++) {
                xb_[i] += a * v[i];
            }
        
        }

    }

}

void GGMLBackend::residual_connection(const Tensor *x, const Tensor *xb2) {
    for (auto i = 0; i < config->dim; i++) {
        ((float*)x->data)[i] += ((float *)xb2->data)[i];
    }
}

void GGMLBackend::silu_hadamard(const Tensor *hb, const Tensor *hb2) {
    for (auto i = 0; i < config->hidden_dim; i++) {
        float val = ((float *)hb->data)[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= ((float *)hb2->data)[i];
        ((float *)hb->data)[i] = val;
    }
}

} // namespace smart