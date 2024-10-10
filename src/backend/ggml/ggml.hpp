
#pragma once
#include "backend/backend.hpp"
#include "backend/ggml/buffer.hpp"
#include "common.hpp"
#include "core/data_type.hpp"
#include "core/tensor.hpp"
#include "ggml.h"
#include "graph/node.hpp"
#include "model/llama-impl/llama_config.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace smart {

namespace ggml {

#define QK8_0 32

struct block_q8_0 {
	uint16_t d;		  // delta
	int8_t qs[QK8_0]; // quants
};

#define QK4_0 32

struct block_q4_0 {
	uint16_t d;			   // delta
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
		return GGML_TYPE_COUNT;
	}
}

static DataType conovrt_datatype_from_ggml(ggml_type tp) {
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
		return DataType::UNKNOWN;
	}
}

static Tensor *convert_from_ggml(ggml_tensor *t) {
	SMART_ASSERT(t != nullptr);
	Tensor::Shape shape;
	Buffer::Stride stride;
	for (int i = 0; i < Tensor::n_dims; i++) {
		shape[i]  = t->ne[i];
		stride[i] = t->nb[i];
	}
	Tensor *opt = new Tensor(conovrt_datatype_from_ggml(t->type), shape);
	Buffer buf	= Buffer(stride, (void *)t->data);
	opt->data	= &buf;
	return opt;
}

static OpTensor convert_to_optensor(const Tensor *t) {
	SMART_ASSERT(t != nullptr);
	OpTensor opt = {
		(void *)t->data,
		ggml::convert_datatype_to_ggml(t->dtype),
	};
	for (int i = 0; i < Tensor::n_dims; i++) {
		opt.ne[i] = t->shape[i];
		opt.nb[i] = ((Buffer *)t->data)->stride[i];
	}

	return opt;
}
} // namespace ggml

// **Note**: Backend receives Tensor not TensorNode
struct GGMLBackend : Backend {
	GGMLBackend(std::shared_ptr<LlamaConfig> config_) : config(config_) {
		wdata  = std::vector<char>(config->dim * 32);
		params = {
			.wsize = (size_t)config->dim * 32,
			.wdata = wdata.data()};
	}

	~GGMLBackend() = default;
	// TODO: How to transfer extra args?
	void matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1);
	void rmsnorm(const Tensor *o, const Tensor *x, const Tensor *weight);
	void softmax(const Tensor *x, int64_t size);
	// void rope(const Tensor *q, const Tensor *k, const int64_t pos); // delete
	void rope(Tensor *q_out, Tensor *k_out, const Tensor *q, const Tensor *k, const Tensor *pos);
	void multihead_attention(const Tensor *q, const Tensor *att, const Tensor *key_cache, const Tensor *val_cache, const Tensor *xb, const int64_t pos, const int64_t L);
	// void residual_connection(const Tensor *x, const Tensor *xb2);
	// void silu_hadamard(const Tensor *hb, const Tensor *hb2);
	void silu_hadamard(const Tensor *out,const Tensor *hb, const Tensor *hb2);
	void add(const Tensor *dst, const Tensor *src0, const Tensor *src1);

private:
	op_compute_params params;
	std::vector<char> wdata;
	void rmsnorm_internal(float *o, float *x, float *weight, int64_t size);
	void softmax_internal(float *x, int64_t size);
	std::shared_ptr<LlamaConfig> config;
};

} // namespace smart
