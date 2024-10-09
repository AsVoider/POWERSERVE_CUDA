
#pragma once
#include "backend/backend.hpp"
#include "common.hpp"
#include "ggml.h"
#include "graph/node.hpp"
#include "model/llama-impl/llama_config.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace smart {

namespace ggml {

#define QK8_0 32

struct block_q8_0{
    uint16_t d;        // delta
    int8_t   qs[QK8_0]; // quants
};

#define QK4_0 32

struct block_q4_0{
    uint16_t d;           // delta
    uint8_t  qs[QK4_0 / 2]; // nibbles / quants
};

void dequantize_row_q8_0(const block_q8_0 *x, float * y, int64_t k);
void dequantize_row_q4_0(const block_q4_0 * x, float * y, int64_t k);

static ggml_type convert_datatype_to_ggml(DataType dtp) {
	return ggml_type(static_cast<std::underlying_type_t<DataType>>(dtp));
}

static DataType conovrt_datatype_from_ggml(ggml_type tp) {
	return DataType(tp);
}

static Tensor *convert_from_ggml(ggml_tensor *t) {
	SMART_ASSERT(t != nullptr);
	Tensor *opt = new Tensor();
	opt->data = (float *)t->data;
	opt->dtype = conovrt_datatype_from_ggml(t->type);
	for (int i = 0; i < GGML_MAX_DIMS; i++) {
		opt->ne[i] = t->ne[i];
		opt->nb[i] = t->nb[i];
	}
	return opt;
}
} // namespace ggml

class GGMLBackend : public Backend {
private:
	op_compute_params params;
	std::vector<char> wdata;
	void rmsnorm_internal(float *o, float *x, float *weight, int64_t size);
	void softmax_internal(float *x, int64_t size);
	std::shared_ptr<LlamaConfig> config;

public:
	GGMLBackend(std::shared_ptr<LlamaConfig> config_) : config(config_) {
		wdata  = std::vector<char>(config->dim * 32);
		params = {
			.wsize = (size_t)config->dim * 32,
			.wdata = wdata.data()
		};
	}

	~GGMLBackend() = default;
	// TODO: How to transfer extra args?
	void matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1);
	void rmsnorm(const Tensor *o, const Tensor *x, const Tensor *weight);
	void softmax(const Tensor *x, int64_t size);
	void rope(const Tensor *q, const Tensor *k, const int64_t pos);
	void multihead_attention(const Tensor *q, const Tensor *att, const Tensor *key_cache, const Tensor *val_cache, const Tensor *xb, const int64_t pos, const int64_t L);
	void residual_connection(const Tensor* x, const Tensor *xb2);
	void silu_hadamard(const Tensor *hb, const Tensor *hb2);
};

} // namespace smart