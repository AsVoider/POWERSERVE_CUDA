#include "tensor.hpp"
#include <cmath>

namespace smart {

OpTensor *get_optensor_from_ggml(ggml_tensor *t) {
    assert(t != nullptr);
    OpTensor *opt = new OpTensor({
        .data = t->data,
        .type = t->type
    });
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        opt->ne[i] = t->ne[i];
        opt->nb[i] = t->nb[i];
    }
    return opt;
}

void free_optensor_deep(OpTensor *opt) {
    if (opt->data != nullptr) {
        switch (opt->type) {
            case GGML_TYPE_F32: delete[] (float *)(opt->data); break;
            default: break;
        }
    }
    opt->data = nullptr;
    delete opt;
}

void free_optensor(OpTensor *opt) {
    delete opt;
}

void dequantize_row_q8_0(const block_q8_0 *x, float * y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

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

    assert(k % qk == 0);

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

void rmsnorm_internal(float* o, float* x, float* weight, int64_t size) {
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

void rmsnorm(OpTensor *o, OpTensor *x, OpTensor *weight) {
    assert(o != nullptr && o->type == GGML_TYPE_F32 && o->data != nullptr);
    assert(x != nullptr && x->type == GGML_TYPE_F32 && x->data != nullptr);
    assert(weight != nullptr && weight->type == GGML_TYPE_F32 && weight->data != nullptr);

    auto size = x->ne[0];

    rmsnorm_internal((float *)o->data, (float *)x->data, (float *)weight->data, size);
}

void softmax_internal(float* x, int64_t size) {
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

void softmax(OpTensor *x, int64_t size) {
    assert(x != nullptr && x->type == GGML_TYPE_F32 && x->data != nullptr);
    softmax_internal((float *)x->data, size);
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

} // namespace smart