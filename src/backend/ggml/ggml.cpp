#include "ggml.hpp"
#include "backend/ggml/buffer.hpp"
#include "ggml.h"
#include "graph/node.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace smart::ggml {

void dequantize_row_q8_0(const block_q8_0 *x, float *y, int64_t k) {
	static const int qk = QK8_0;

	SMART_ASSERT(k % qk == 0);

	const int nb = k / qk;

	for (int i = 0; i < nb; i++) {
		const float d = ggml_fp16_to_fp32(x[i].d);

		for (int j = 0; j < qk; ++j) {
			y[i * qk + j] = x[i].qs[j] * d;
		}
	}
}

void dequantize_row_q4_0(const block_q4_0 *x, float *y, int64_t k) {
	static const int qk = QK4_0;

	SMART_ASSERT(k % qk == 0);

	const int nb = k / qk;

	for (int i = 0; i < nb; i++) {
		const float d = ggml_fp16_to_fp32(x[i].d);

		for (int j = 0; j < qk / 2; ++j) {
			const int x0 = (x[i].qs[j] & 0x0F) - 8;
			const int x1 = (x[i].qs[j] >> 4) - 8;

			y[i * qk + j + 0]	   = x0 * d;
			y[i * qk + j + qk / 2] = x1 * d;
		}
	}
}

void GGMLBackend::matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
	// W (d,n) @ x (n,) -> xout (d,)
	// by far the most amount of time is spent inside this little function
	OpTensor dst_ = ggml::convert_to_optensor(dst);
	OpTensor src0_ = ggml::convert_to_optensor(src0);
	OpTensor src1_ = ggml::convert_to_optensor(src1);

	ggml_compute_forward_op_mul_mat(&params, &dst_, &src0_, &src1_);
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

void GGMLBackend::rmsnorm(const Tensor *o, const Tensor *x, const Tensor *weight) const {
	auto size = x->shape[0];

	rmsnorm_internal((float *)o->get<Buffer>().data, (float *)x->get<Buffer>().data, (float *)weight->get<Buffer>().data, size);
}

void GGMLBackend::softmax_internal(float *out_, float *x_, size_t size) const {
	// find max value (for numerical stability)
	float max_val = x_[0];
	for (int i = 1; i < size; i++) {
		if (x_[i] > max_val) {
			max_val = x_[i];
		}
	}
	// exp and sum
	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		out_[i] = expf(x_[i] - max_val);
		sum += out_[i];
	}
	// normalize
	for (int i = 0; i < size; i++) {
		out_[i] /= sum;
	}
}

void GGMLBackend::softmax(const Tensor *out, const Tensor *x) const {
	softmax_internal((float *)out->get<Buffer>().data, (float *)x->get<Buffer>().data, x->shape[0]);
}

// TODO: Rope's pos should be a tensor and we need rope_base (llama2 = 10000, llama3 = 300000 ...)
void GGMLBackend::rope(Tensor *q_out, Tensor *k_out, const Tensor *q, const Tensor *k, const Tensor *pos) const {
	auto dim = q->shape[0];
	auto head_size = dim / config->n_heads;
	auto kv_dim	   = (config->dim * config->n_kv_heads) / config->n_heads;
	auto pos_data = (int32_t *)pos->get<Buffer>().data;

	for (int i = 0; i < dim; i += 2) {
		int head_dim = i % head_size;
		float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
		float val	 = pos_data[0] * freq;
		float fcr = cosf(val);
		float fci = sinf(val);
		int rotn	 = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
		for (int v = 0; v < rotn; v++) {
			float *vec = v == 0 ? (float *)q_out->get<Buffer>().data : (float *)k_out->get<Buffer>().data;
			float v0 = vec[i];
			float v1 = vec[i + 1];
			vec[i] = v0 * fcr - v1 * fci;
			vec[i + 1] = v0 * fci + v1 * fcr;
		}
	}
}

// TODO: Need to be split
// prev: q, att, key_cache, val_cache, xb, pos, L; next: {att, xb}
void GGMLBackend::multihead_attention(const Tensor *out, const Tensor *q, const Tensor *key_cache, const Tensor *val_cache, const Tensor *pos, const int64_t L) const {
	auto dim	   = config->dim;
	auto kv_dim	   = (config->dim * config->n_kv_heads) / config->n_heads;
	auto kv_mul	   = config->n_heads / config->n_kv_heads;
	auto head_size = dim / config->n_heads;
	uint64_t loff  = L * config->seq_len * kv_dim;
	auto pos_data = (int32_t *)pos->get<Buffer>().data;
	auto att = std::vector<float>(config->n_heads * config->seq_len);

	uint32_t h = 0;
	for (h = 0; h < config->n_heads; h++) {
		auto q_	  = (float *)q->get<Buffer>().data + h * head_size;
		auto att_ = att.data() + h * config->seq_len;

		for (auto t = 0; t <= pos_data[0]; t++) {
			auto k	   = (float *)key_cache->get<Buffer>().data + loff + t * kv_dim + (h / kv_mul) * head_size;
			auto score = 0.0f;

			for (auto i = 0; i < head_size; i++) {
				score += q_[i] * k[i];
			}

			score /= sqrtf(head_size);
			att_[t] = score;
		}

		softmax_internal(att_, att_, pos_data[0] + 1);

		auto xb_ = (float *)out->get<Buffer>().data + h * head_size;
		memset(xb_, 0, head_size * sizeof(float));

		for (auto t = 0; t <= pos_data[0]; t++) {
			auto v = (float *)val_cache->get<Buffer>().data + loff + t * kv_dim + (h / kv_mul) * head_size;
			auto a = att_[t];

			for (auto i = 0; i < head_size; i++) {
				xb_[i] += a * v[i];
			}
		}
	}
}

void GGMLBackend::add(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
	for (auto i = 0; i < config->dim; i++) {
		((float *)dst->get<Buffer>().data)[i] = ((float *)src0->get<Buffer>().data)[i] + ((float *)src1->get<Buffer>().data)[i];
	}
}
void GGMLBackend::silu_hadamard(const Tensor *out,const Tensor *hb, const Tensor *hb2) const {
	for (auto i = 0; i < config->hidden_dim; i++) {
		float val = ((float *)hb->get<Buffer>().data)[i];
		// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
		val *= (1.0f / (1.0f + expf(-val)));
		// elementwise multiply with w3(x)
		val *= ((float *)hb2->get<Buffer>().data)[i];
		((float *)out->get<Buffer>().data)[i] = val;
	}
}

auto GGMLBackend::create_float_buffer(Tensor::Shape shape) -> BufferPtr {
	Buffer::Stride stride;
	stride[0] = sizeof(float);
	for (size_t i = 1; i < shape.size(); i++) {
		stride[i] = stride[i - 1] * shape[i - 1];
	}

	size_t size = stride.back() * shape.back();

	return std::make_shared<Buffer>(stride, malloc(size), true);
}

} // namespace smart
