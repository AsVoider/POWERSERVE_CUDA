#pragma once

#include "backend/ggml/ggml.hpp"
#include "core/tensor.hpp"
#include "fmt/base.h"
#include "graph/graph.hpp"
#include "model/llama/llama_config.hpp"
#include "model/llama/llama_weight.hpp"
#include <cstdlib>
#include <vector>
namespace smart {

struct Attention {

	std::shared_ptr<LlamaConfig> config;
	std::shared_ptr<LlamaWeight> weights;

	Tensor gkey_cache;			   // kv_dim x n_kv_heads x seq_len
	Tensor gval_cache;			   // kv_dim x n_kv_heads x seq_len

	Attention(std::shared_ptr<LlamaConfig> config, std::shared_ptr<LlamaWeight> weights)
		: config(config), weights(weights),
		  gkey_cache(DataType::FP32, {(config->dim * config->n_kv_heads) / config->n_heads, config->seq_len, config->n_layers}),
		  gval_cache(DataType::FP32, {(config->dim * config->n_kv_heads) / config->n_heads, config->seq_len, config->n_layers}) {
		// FIXME: Too Aggressive to allocate memory
		ggml::GGMLBackend backend(config); // tmp
		Tensor::Shape shape = {(config->dim * config->n_kv_heads) / config->n_heads, config->seq_len, config->n_layers, 1};
		auto kb				= backend.create_buffer<float>(shape);
		auto vb				= backend.create_buffer<float>(shape);
		gkey_cache.data		= kb;
		gval_cache.data		= vb;
	}

	TensorNode *build(Graph &g, TensorNode *x, int64_t L, TensorNode *pos_tensor, int32_t pos);

};

} // namespace smart
