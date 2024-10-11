#pragma once

#include "backend/ggml/buffer.hpp"
#include "backend/ggml/ggml.hpp"
#include "core/data_type.hpp"
#include <vector>
namespace smart {
struct LayerWeights {
	Tensor attn_norm; // "blk.$.attn_norm.weight" (layer, dim)
	Tensor ffn_norm;  // "blk.$.ffn_norm.weight" (layer, dim)
	// dim == n_heads * head_size
	Tensor attn_q;		// "blk.$.attn_q.weight" (layer, dim, n_heads * head_size)
	Tensor attn_k;		// "blk.$.attn_k.weight" (layer, dim, n_kv_heads * head_size)
	Tensor attn_v;		// "blk.$.attn_v.weight" (layer, dim, n_kv_heads * head_size)
	Tensor attn_output; // "blk.$.attn_output.weight" (layer, n_heads * head_size, dim)

	Tensor ffn_gate; // "blk.$.ffn_gate.weight" (layer, dim, hidden_dim)
	Tensor ffn_up;	 // "blk.$.ffn_up.weight" (layer, dim, hidden_dim)
	Tensor ffn_down; // "blk.$.ffn_down.weight" (layer, hidden_dim, dim)

	LayerWeights(ggml_context *ctx, uint32_t layer)
		: attn_norm(get_tensor(ctx, layer, "attn_norm.weight")),
		  ffn_norm(get_tensor(ctx, layer, "ffn_norm.weight")),
		  attn_q(get_tensor(ctx, layer, "attn_q.weight")),
		  attn_k(get_tensor(ctx, layer, "attn_k.weight")),
		  attn_v(get_tensor(ctx, layer, "attn_v.weight")),
		  attn_output(get_tensor(ctx, layer, "attn_output.weight")),
		  ffn_gate(get_tensor(ctx, layer, "ffn_gate.weight")),
		  ffn_up(get_tensor(ctx, layer, "ffn_up.weight")),
		  ffn_down(get_tensor(ctx, layer, "ffn_down.weight")) {}

	static Tensor get_tensor(ggml_context *ctx, uint32_t layer, const char *name) {
		std::string tensor_name = fmt::format("blk.{}.{}", layer, name);
		ggml_tensor *t			= ggml_get_tensor(ctx, tensor_name.c_str());
		if (t == nullptr) {
			throw std::runtime_error(fmt::format("Failed to get tensor: {}", tensor_name));
		}
		return ggml::convert_from_ggml(t);
	}

	~LayerWeights() = default;
};

struct LlamaWeight {
	Tensor token_embedding_table;		// "token_embd.weight" (vocab_size, dim)
	Tensor output_weight;				// "output.weight" (vocab_size, dim)
	Tensor rms_final_weight;			// "output_norm.weight" (dim,)
	std::vector<float> fp32_embd_table; // quantized embedding table

	std::vector<LayerWeights> lw;

	LlamaWeight(ggml_context *ctx, uint32_t n_layers, uint32_t dim)
		: token_embedding_table(ggml::convert_from_ggml(ggml_get_tensor(ctx, "token_embd.weight"))),
		  output_weight(ggml::convert_from_ggml(ggml_get_tensor(ctx, "output.weight"))),
		  rms_final_weight(ggml::convert_from_ggml(ggml_get_tensor(ctx, "output_norm.weight"))) {
		auto embedding	= ggml_get_tensor(ctx, "token_embd.weight");
		fp32_embd_table = std::vector<float>(ggml_nelements(embedding)); // + 2G

		switch (token_embedding_table.dtype_) {
		case DataType::FP32:
			std::memcpy(fp32_embd_table.data(), embedding->data, ggml_nelements(embedding) * sizeof(float));
			break;
		case DataType::GGML_Q4_0:
			ggml::dequantize_row_q4_0((ggml::block_q4_0 *)token_embedding_table.get<ggml::Buffer>().data, fp32_embd_table.data(), ggml_nelements(embedding));
			break;
		case DataType::GGML_Q8_0:
			ggml::dequantize_row_q8_0((ggml::block_q8_0 *)token_embedding_table.get<ggml::Buffer>().data, fp32_embd_table.data(), ggml_nelements(embedding));
			break;
		default:
			break;
		}
		// init layers' weights
		{
			for (int layer = 0; layer < n_layers; layer++) {
				lw.push_back(LayerWeights(ctx, layer));
			}
		}
	}
	~LlamaWeight() = default;
};
} // namespace smart