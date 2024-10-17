#pragma once

#include "backend/backend.hpp"
#include "backend/ggml/buffer.hpp"
#include "common.hpp"
#include "core/data_type.hpp"
#include "core/tensor.hpp"
#include "ggml.h"
#include "model/llama/llama_config.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace smart::ggml {

#define QK8_0 32

struct block_q8_0 {
    uint16_t d;       // delta
    int8_t qs[QK8_0]; // quants
};

#define QK4_0 32

struct block_q4_0 {
    uint16_t d;            // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
};

void dequantize_row_q8_0(const block_q8_0 *x, float *y, int64_t k);
void dequantize_row_q4_0(const block_q4_0 *x, float *y, int64_t k);

static ggml_type convert_datatype_to_ggml(DataType dtp) {
    switch (dtp) {
    case DataType::FP32:
        return GGML_TYPE_F32;
    case DataType::FP16:
        return GGML_TYPE_F16;
    case DataType::GGML_Q4_0:
        return GGML_TYPE_Q4_0;
    case DataType::GGML_Q8_0:
        return GGML_TYPE_Q8_0;
    default:
        SMART_ASSERT(false);
    }
}

static DataType convert_datatype_from_ggml(ggml_type tp) {
    switch (tp) {
    case GGML_TYPE_F32:
        return DataType::FP32;
    case GGML_TYPE_F16:
        return DataType::FP16;
    case GGML_TYPE_Q4_0:
        return DataType::GGML_Q4_0;
    case GGML_TYPE_Q8_0:
        return DataType::GGML_Q8_0;
    default:
        SMART_ASSERT(false);
    }
}

static Tensor convert_from_ggml(ggml_tensor *t) {
    SMART_ASSERT(t != nullptr);
    Tensor::Shape shape;
    Buffer::Stride stride;
    for (size_t i = 0; i < Tensor::max_n_dims; i++) {
        shape[i]  = t->ne[i];
        stride[i] = t->nb[i];
    }
    Tensor tensor(convert_datatype_from_ggml(t->type), shape);
    tensor.m_data = std::make_shared<Buffer>(stride, t->data);
    return tensor;
}

static OpTensor convert_to_optensor(const Tensor *t) {
    SMART_ASSERT(t != nullptr);

    OpTensor opt;
    opt.data = t->get<ggml::Buffer>().m_data;
    opt.type = convert_datatype_to_ggml(t->m_dtype);
    for (size_t i = 0; i < Tensor::max_n_dims; i++) {
        opt.ne[i] = t->m_shape[i];
        opt.nb[i] = t->get<ggml::Buffer>().m_stride[i];
    }

    return opt;
}

// **Note**: Backend receives Tensor not TensorNode
struct GGMLBackend : Backend {
private:
    op_compute_params m_params;
    std::vector<char> m_wdata;
    std::shared_ptr<LlamaConfig> m_config;

public:
    explicit GGMLBackend(std::shared_ptr<LlamaConfig> config) : m_wdata(config->dim * 32), m_config(config) {
        m_params = {
            .wsize = (size_t)config->dim * 32,
            .wdata = m_wdata.data(),
        };
    }

    ~GGMLBackend() override = default;

public:
    void matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const;
    void rmsnorm(const Tensor *o, const Tensor *x, const Tensor *weight) const;
    void softmax(const Tensor *out, const Tensor *x) const;
    void rope(Tensor *q_out, Tensor *k_out, const Tensor *q, const Tensor *k, const Tensor *pos) const;
    void multihead_attention(
        const Tensor *out,
        const Tensor *q,
        const Tensor *key_cache,
        const Tensor *val_cache,
        const Tensor *pos,
        const int64_t L
    ) const;
    void silu_hadamard(const Tensor *out, const Tensor *hb, const Tensor *hb2) const;
    void add(const Tensor *dst, const Tensor *src0, const Tensor *src1) const;
    void copy(const Tensor *dst, const Tensor *src, const int64_t off) const;

public:
    template <typename T>
    auto create_buffer(Tensor::Shape shape) -> BufferPtr {
        Buffer::Stride stride;
        stride[0] = sizeof(T);
        for (size_t i = 1; i < shape.size(); i++) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        size_t size = stride.back() * shape.back();

        return std::make_shared<Buffer>(stride, malloc(size), true);
    }

private:
    void rmsnorm_internal(float *o, float *x, float *weight, int64_t size) const;
    void softmax_internal(float *out, float *x, size_t size) const;
};

} // namespace smart::ggml
