#include "llama_model.hpp"
#include "backend/platform.hpp"
#include "common.hpp"
#include "fmt/base.h"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "graph/sched.hpp"
#include "model/llama-impl/llama_buffer.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
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
	{
		config = std::make_shared<LlamaConfig>(gguf_ctx);
	}
	// prepare weights (+ 2G)
	{
		weights = std::make_shared<LlamaWeight>(ggml_ctx, config->n_layers);
	}
	// prepare global buffer (+ 33G)
	{
		buffer = std::make_shared<LlamaBuffer>(config);
	}
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
	{
		// prepare input : embeding token tensor [dim,]
		SMART_ASSERT(token * dim + dim <= weights->fp32_embd_table.size());
		std::copy(
			weights->fp32_embd_table.begin() + token * dim,
			weights->fp32_embd_table.begin() + (token + 1) * dim,
			buffer->x.begin());
	}
	// attention and ffn
	{
		for (auto L = 0; L < config->n_layers; L++) {
			attn.build_graph(g, config, weights, buffer, L, pos);
			ffn.build_graph(g, config, weights, buffer, L, pos);
		}
	}

	// final output
	{
		auto final_rms_norm = std::make_shared<Operator>(OpType::OP_RMS_NORM);
		auto xi				= buffer->get_new_node_dim("x");
		auto xo				= buffer->get_new_node_dim("x");
		g.nodes.push_back(final_rms_norm);
		g.nodes.push_back(xi);
		g.nodes.push_back(xo);
		add_input(final_rms_norm, xi);
		add_input(final_rms_norm, weights->rms_final_weight);
		add_output(final_rms_norm, xo);

		auto malmut_log = std::make_shared<Operator>(OpType::OP_MUL_MAT);
		auto logits		= buffer->get_new_node_logits();
		g.nodes.push_back(malmut_log);
		g.nodes.push_back(logits);
		add_input(malmut_log, xo);
		add_input(malmut_log, weights->output_weight);
		add_output(malmut_log, logits);
	}
	Sched sched;
	Platform plat(config);
	sched.run(g, plat);
	return buffer->logits;
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