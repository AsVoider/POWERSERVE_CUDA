#include "llama_model.hpp"
#include "backend/ggml/buffer.hpp"
#include "backend/platform.hpp"
#include "common.hpp"
#include "executor/executor.hpp"
#include "fmt/base.h"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"
#include <cstring>
#include <string>
#include <vector>

namespace smart {

LlamaModel::LlamaModel(std::string filename_) {
	filename = filename_;
	// load file meta data (+ 4G)
	{
		gguf_init_params params = {
			.no_alloc = false,
			.ctx	  = &ggml_ctx};
		gguf_ctx = gguf_init_from_file(filename.c_str(), params);
		SMART_ASSERT(gguf_ctx != nullptr);
		SMART_ASSERT(ggml_ctx != nullptr);
	}
	// prepare data
	config = std::make_shared<LlamaConfig>(gguf_ctx);
	// prepare weights (+ 2G)
	weights = std::make_shared<LlamaWeight>(ggml_ctx, config->n_layers, config->dim);
	// modules
	attn = std::make_shared<Attention>(config, weights);
	ffn	 = std::make_shared<FFN>(config, weights);
}

LlamaModel::~LlamaModel() {
	gguf_free(gguf_ctx);
}

Graph *LlamaModel::prefill() {
	return nullptr;
}

Graph *LlamaModel::decode() {
	return nullptr;
}

std::vector<float> LlamaModel::forward(int token, int pos) {
	Graph g;
	auto dim = config->dim;

	// input embedding
	// prepare input : embeding token tensor [dim,]
	SMART_ASSERT(token * dim + dim <= weights->fp32_embd_table.size());
	auto x					= g.new_tensor(DataType::FP32, {dim});
	TensorNode *tensor_embd = x;
	auto pos_tensor			= g.new_tensor(DataType::INT32, {1});
	// attention and ffn
	for (auto L = 0; L < config->n_layers; L++) {
		auto att_o = attn->build(g, x, L, pos_tensor, pos);
		auto ffn_o = ffn->build(g, att_o, L);
		x		   = ffn_o;
	}

	// final output
	auto rms_final_w	= g.add_tensor(weights->rms_final_weight);
	auto final_rms_norm = g.rms_norm(x, rms_final_w);

	auto output_w = g.add_tensor(weights->output_weight);
	auto logits	  = g.mat_mul(final_rms_norm, output_w);

	Platform plat(config);
	Executor executor(plat, g);
	executor.allocate_buffers();
	memcpy(tensor_embd->get<ggml::Buffer>().data_, (void *)(weights->fp32_embd_table.data() + token * dim), dim * sizeof(float));
	((int32_t *)pos_tensor->get<ggml::Buffer>().data_)[0] = pos;

	executor.run();
	float *logits_data = (float *)(logits->get<ggml::Buffer>().data_);

	return std::vector<float>(logits_data, logits_data + dim);
}

void LlamaModel::generate(Tokenizer *tk, Sampler *sampler, std::string prompt, int steps) {
	// encode the (string) prompt into tokens sequence
	int num_prompt_tokens = 0;
	auto prompt_tokens	  = tk->tokenize(prompt, true);
	num_prompt_tokens	  = prompt_tokens.size();

	SMART_ASSERT(num_prompt_tokens >= 1);
	// start the main loop
	long start = 0;				   // used to time our code, only initialized after first iteration
	int next;					   // will store the next token in the sequence
	auto token = prompt_tokens[0]; // kick off with the first token in the prompt
	int pos	   = 0;				   // position in the sequence
	while (pos < steps) {

		// forward the transformer to get logits for the next token
		// float* logits = forward(token, pos);
		std::vector<float> logits = forward(token, pos);

		// advance the state machine
		if (pos < num_prompt_tokens - 1) {
			// TODO: prefill
			// if we are still processing the input prompt, force the next prompt token
			next = prompt_tokens[pos + 1];
		} else {
			// TODO: Decode
			// otherwise sample the next token from the logits
			next = sampler->sample(logits);
		}
		pos++;

		// data-dependent terminating condition: the BOS token delimits sequences
		if (next == tk->bos_token()) {
			break;
		}

		// print the token as string, decode it with the Tokenizer object
		auto piece = tk->to_string(next);
		fmt::print("{}", piece);
		fflush(stdout);
		token = next;

		// init the timer here because the first iteration can be slower
		if (start == 0) {
			start = time_in_ms();
		}
	}
	fmt::println("");

	if (pos > 1) {
		long end = time_in_ms();
		fmt::println(stderr, "achieved tok/s: {}\n", (pos - 1) / (double)(end - start) * 1000);
	}
}

} // namespace smart