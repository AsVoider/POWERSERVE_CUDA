#pragma once

#include "common.hpp"
#include "fmt/base.h"
#include "graph/node.hpp"
#include "model/llama-impl/llama_config.hpp"
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
namespace smart {

struct LlamaBuffer {

	std::shared_ptr<LlamaConfig> config;
	// use array for fixed list
	std::vector<float> x;
	std::vector<float> xb;
	std::vector<float> xb2;
	std::vector<float> hb;
	std::vector<float> hb2;
	std::vector<float> q;
	// std::vector<float> k;
	// std::vector<float> v;
	std::vector<float> att;
	std::vector<float> logits;
	std::vector<float> key_cache;
	std::vector<float> value_cache;

	LlamaBuffer(std::shared_ptr<LlamaConfig> config_) : config(config_) {
		uint64_t kv_dim		= (config->dim * config->n_kv_heads) / config->n_heads;
		uint64_t dim		= config->dim;
		uint64_t hidden_dim = config->hidden_dim;
		uint64_t n_layers	= config->n_layers;
		uint64_t large_size = n_layers * config->seq_len * kv_dim;

		x = std::vector<float>(dim);

		xb	= std::vector<float>(dim);
		xb2 = std::vector<float>(dim);

		hb	= std::vector<float>(hidden_dim);
		hb2 = std::vector<float>(hidden_dim);

		q = std::vector<float>(dim);

		att	   = std::vector<float>(config->n_heads * config->seq_len);
		logits = std::vector<float>(config->vocab_size);
		key_cache.reserve(large_size);
		value_cache.reserve(large_size);
		// key_cache	= std::vector<float>(large_size);  // + 16G
		// value_cache = std::vector<float>(large_size);  // + 16G
	}

	~LlamaBuffer() = default;

	std::shared_ptr<Tensor> get_new_node_dim(std::string name) {
		float *data = nullptr;
		if (name == "x")
			data = x.data();
		else if (name == "xb")
			data = xb.data();
		else if (name == "xb2")
			data = xb2.data();
		else if (name == "q")
			data = q.data();
		else
			SMART_ASSERT(false);

		return std::make_shared<Tensor>(Tensor{
			data,
			{(int64_t)config->dim, 1, 1, 1},
			{sizeof(float), sizeof(float) * config->dim, sizeof(float) * config->dim, sizeof(float) * config->dim}});
	}

	std::shared_ptr<Tensor> get_new_node_hidden_dim(std::string name) {
		float *data = nullptr;
		if (name == "hb")
			data = hb.data();
		else if (name == "hb2")
			data = hb2.data();
		else
			SMART_ASSERT(false);

		return std::make_shared<Tensor>(Tensor{
			data,
			{config->hidden_dim, 1, 1, 1},
			{sizeof(float), sizeof(float) * config->hidden_dim, sizeof(float) * config->hidden_dim, sizeof(float) * config->hidden_dim}});
	}

	std::shared_ptr<Tensor> get_new_node_att() {
		float *data = att.data();

		return std::make_shared<Tensor>(Tensor{
			data,
			{config->seq_len, config->n_heads, 1, 1},
			{sizeof(float), sizeof(float) * config->seq_len, sizeof(float) * config->n_heads * config->seq_len, sizeof(float) * config->n_heads * config->seq_len}});
	}

	std::shared_ptr<Tensor> get_new_node_logits() {
		float *data = logits.data();

		return std::make_shared<Tensor>(Tensor{
			data,
			{(int64_t)config->vocab_size, 1, 1, 1},
			{sizeof(float), sizeof(float) * config->vocab_size, sizeof(float) * config->vocab_size, sizeof(float) * config->vocab_size}});
	}

	std::shared_ptr<Tensor> get_new_node_cache(std::string name) {
		float *data		= nullptr;
		uint64_t kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
		if (name == "key")
			data = key_cache.data();
		else if (name == "value")
			data = value_cache.data();
		else
			SMART_ASSERT(false);

		return std::make_shared<Tensor>(Tensor{
			data,
			{(int64_t)kv_dim, config->seq_len, config->n_layers, 1},
			{sizeof(float), sizeof(float) * kv_dim, sizeof(float) * kv_dim * config->seq_len, sizeof(float) * config->n_layers * config->seq_len * kv_dim}});
	}

	std::shared_ptr<Tensor> get_new_node_kv_dim(std::string name, uint64_t off) {
		float *data		= nullptr;
		uint64_t kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
		if (name == "k")
			data = key_cache.data() + off;
		else if (name == "v")
			data = value_cache.data() + off;
		else
			SMART_ASSERT(false);

		return std::make_shared<Tensor>(Tensor{
			data,
			{(int64_t)kv_dim, 1, 1, 1},
			{sizeof(float), sizeof(float) * kv_dim, sizeof(float) * kv_dim, sizeof(float) * kv_dim}});
	}

	std::shared_ptr<Tensor> get_new_node_int64(int64_t val) {

		auto t							  = std::make_shared<Tensor>(Tensor{
			   1 * sizeof(int64_t),
									   {1, 1, 1, 1},
									   {sizeof(int64_t), sizeof(int64_t), sizeof(int64_t), sizeof(int64_t)},
			   DataType::I64});
		*((int64_t *)t->container.data()) = val;
		return t;
	}
};

} // namespace smart