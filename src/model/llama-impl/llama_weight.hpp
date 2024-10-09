#pragma once

#include "backend/ggml/ggml.hpp"
#include "graph/node.hpp"
#include <memory>
namespace smart {
struct LayerWeights {
	std::shared_ptr<Tensor> attn_norm; // "blk.$.attn_norm.weight" (layer, dim)
	std::shared_ptr<Tensor> ffn_norm;	 // "blk.$.ffn_norm.weight" (layer, dim)
	// dim == n_heads * head_size
	std::shared_ptr<Tensor> attn_q;	   // "blk.$.attn_q.weight" (layer, dim, n_heads * head_size)
	std::shared_ptr<Tensor> attn_k;	   // "blk.$.attn_k.weight" (layer, dim, n_kv_heads * head_size)
	std::shared_ptr<Tensor> attn_v;	   // "blk.$.attn_v.weight" (layer, dim, n_kv_heads * head_size)
	std::shared_ptr<Tensor> attn_output; // "blk.$.attn_output.weight" (layer, n_heads * head_size, dim)

	std::shared_ptr<Tensor> ffn_gate; // "blk.$.ffn_gate.weight" (layer, dim, hidden_dim)
	std::shared_ptr<Tensor> ffn_up;	// "blk.$.ffn_up.weight" (layer, dim, hidden_dim)
	std::shared_ptr<Tensor> ffn_down; // "blk.$.ffn_down.weight" (layer, hidden_dim, dim)

	LayerWeights(ggml_context *ctx, uint32_t layer) {
		attn_norm   = std::shared_ptr<Tensor>(ggml::convert_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_norm.weight", layer).c_str())));
		ffn_norm    = std::shared_ptr<Tensor>(ggml::convert_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.ffn_norm.weight", layer).c_str())));
		attn_q      = std::shared_ptr<Tensor>(ggml::convert_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_q.weight", layer).c_str())));
		attn_k      = std::shared_ptr<Tensor>(ggml::convert_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_k.weight", layer).c_str())));
		attn_v      = std::shared_ptr<Tensor>(ggml::convert_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_v.weight", layer).c_str())));
		attn_output = std::shared_ptr<Tensor>(ggml::convert_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_output.weight", layer).c_str())));
		ffn_gate    = std::shared_ptr<Tensor>(ggml::convert_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.ffn_gate.weight", layer).c_str())));
		ffn_up      = std::shared_ptr<Tensor>(ggml::convert_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.ffn_up.weight", layer).c_str())));
		ffn_down    = std::shared_ptr<Tensor>(ggml::convert_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.ffn_down.weight", layer).c_str())));
	} 

	~LayerWeights() = default;

};

struct LlamaWeight {
	std::shared_ptr<Tensor> token_embedding_table; // "token_embd.weight" (vocab_size, dim)
	std::shared_ptr<Tensor> output_weight;         // "output.weight" (vocab_size, dim)
	std::shared_ptr<Tensor> rms_final_weight;      // "output_norm.weight" (dim,)
	std::vector<float> fp32_embd_table;      // quantized embedding table

	std::vector<LayerWeights> lw;

	LlamaWeight(ggml_context *ctx, uint32_t n_layers) {
		auto embedding = ggml_get_tensor(ctx, "token_embd.weight");
		token_embedding_table = std::shared_ptr<Tensor>(ggml::convert_from_ggml(embedding));
		output_weight         = std::shared_ptr<Tensor>(ggml::convert_from_ggml(ggml_get_tensor(ctx, "output.weight")));
		rms_final_weight      = std::shared_ptr<Tensor>(ggml::convert_from_ggml(ggml_get_tensor(ctx, "output_norm.weight")));
		fp32_embd_table = std::vector<float>(ggml_nelements(embedding)); // + 2G

		switch (token_embedding_table->dtype) {
			case DataType::F32:
				std::memcpy(fp32_embd_table.data(), embedding->data, ggml_nelements(embedding) * sizeof(float));
				break;
			case DataType::Q4_0:
				ggml::dequantize_row_q4_0((ggml::block_q4_0 *)token_embedding_table->data, fp32_embd_table.data(), ggml_nelements(embedding));
				break;
			case DataType::Q8_0:
				ggml::dequantize_row_q8_0((ggml::block_q8_0 *)token_embedding_table->data, fp32_embd_table.data(), ggml_nelements(embedding));
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
}