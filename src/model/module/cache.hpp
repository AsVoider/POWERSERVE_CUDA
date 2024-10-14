#pragma once
#include "backend/ggml/ggml.hpp"
#include "core/tensor.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/llama/llama_config.hpp"

namespace smart {

struct Cache {

	Tensor cache;
	std::shared_ptr<LlamaConfig> config;


	void add_cache(Graph &g, TensorNode *tensor, size_t offset);
	TensorNode *add_cache_node(Graph &g);


	Cache(std::shared_ptr<LlamaConfig> config) 
        : config(config), 
          cache(DataType::FP32, {config->dim * config->n_kv_heads / config->n_heads, config->seq_len, config->n_layers}) 
    {
        // FIXME: Too Aggressive to allocate memory
        ggml::GGMLBackend backend(config); // tmp
		Tensor::Shape shape = {(config->dim * config->n_kv_heads) / config->n_heads, config->seq_len, config->n_layers, 1};
		auto buffer				= backend.create_buffer<float>(shape);
		cache.data_	= buffer;
	}
	~Cache() = default;
};

} // namespace smart