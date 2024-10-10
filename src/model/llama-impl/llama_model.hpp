#pragma once

#include "ggml.h"
#include "graph/graph.hpp"
#include "model/llama-impl/llama_buffer.hpp"
#include "model/llama-impl/llama_config.hpp"
#include "model/llama-impl/llama_weight.hpp"
#include "model/model.hpp"
#include "model/module/attention.hpp"
#include "model/module/ffn.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"
#include <cmath>

#include <cstring>
#include <memory>
#include <string>
#include <vector>
namespace smart {

struct LlamaModel : Model {
	Graph *prefill() override;
	Graph *decode() override;
	void generate(Tokenizer *tk, Sampler *sampler, std::string prompt, int steps);
	std::vector<float> forward(int token, int pos);

	LlamaModel(std::string filename_);
	~LlamaModel();

private:
	// ggml need those context
	ggml_context *ggml_ctx;
	gguf_context *gguf_ctx;

	std::shared_ptr<LlamaConfig> config;
	std::shared_ptr<LlamaWeight> weights;
	Attention attn;
	FFN ffn;

	// global buffer
	std::shared_ptr<LlamaBuffer> buffer;
};

} // namespace smart
