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
		auto kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
		for (auto L = 0; L < config->n_layers; L++) {
			// attn.build_graph(g);
			{
				auto rmsnorm_node = std::make_shared<Operator>(OpType::OP_RMS_NORM);
				auto x			  = buffer->get_new_node_dim("x");
				auto att_norm_o	  = buffer->get_new_node_dim("xb");
				g.nodes.push_back(rmsnorm_node);
				g.nodes.push_back(att_norm_o);
				add_input(rmsnorm_node, x);
				add_input(rmsnorm_node, weights->lw[L].attn_norm);
				add_output(rmsnorm_node, att_norm_o);

				// QKV
				auto malmut_q = std::make_shared<Operator>(OpType::OP_MUL_MAT);
				auto q		  = buffer->get_new_node_dim("q");
				g.nodes.push_back(malmut_q);
				g.nodes.push_back(q);
				add_input(malmut_q, att_norm_o);
				add_input(malmut_q, weights->lw[L].attn_q);
				add_output(malmut_q, q);

				uint64_t loff = L * config->seq_len * kv_dim;

				auto malmut_k = std::make_shared<Operator>(OpType::OP_MUL_MAT);
				auto k		  = buffer->get_new_node_kv_dim("k", loff + pos * kv_dim);
				g.nodes.push_back(malmut_k);
				g.nodes.push_back(k);
				add_input(malmut_k, att_norm_o);
				add_input(malmut_k, weights->lw[L].attn_k);
				add_output(malmut_k, k);

				auto malmut_v = std::make_shared<Operator>(OpType::OP_MUL_MAT);
				auto v		  = buffer->get_new_node_kv_dim("v", loff + pos * kv_dim);
				g.nodes.push_back(malmut_v);
				g.nodes.push_back(v);
				add_input(malmut_v, att_norm_o);
				add_input(malmut_v, weights->lw[L].attn_v);
				add_output(malmut_v, v);

				// rope
				auto rope_node = std::make_shared<Operator>(OpType::OP_ROPE);
				auto pos_	   = buffer->get_new_node_int64(pos);

				g.nodes.push_back(rope_node);
				g.nodes.push_back(pos_);
				add_input(rope_node, q);
				add_input(rope_node, k);
				add_input(rope_node, pos_);

				// multihead attention
				auto mha_node  = std::make_shared<Operator>(OpType::OP_MHA);
				auto rot_q	   = buffer->get_new_node_dim("q");
				auto att	   = buffer->get_new_node_att();
				auto mha_xb	   = buffer->get_new_node_dim("xb");
				auto L_		   = buffer->get_new_node_int64(L);
				auto key_cache = buffer->get_new_node_cache("key");
				auto val_cache = buffer->get_new_node_cache("value");

				g.nodes.push_back(mha_node);
				g.nodes.push_back(rot_q);
				g.nodes.push_back(att);
				add_input(mha_node, rot_q);
				add_input(mha_node, att);
				add_input(mha_node, key_cache);
				add_input(mha_node, val_cache);
				add_input(mha_node, mha_xb); // xb
				add_input(mha_node, pos_);
				add_input(mha_node, L_);

				// if (L == 1) {
				// 	fmt::println("L: {}; data: {}", L, *((int64_t *)L_->container.data()));
				// 	Sched sched;
				// 	Platform plat(config);
				// 	sched.run(g, plat);

				// }
				auto malmut_attn = std::make_shared<Operator>(OpType::OP_MUL_MAT);
				auto attn_o		 = buffer->get_new_node_dim("xb2");
				g.nodes.push_back(malmut_attn);
				g.nodes.push_back(attn_o);
				add_input(malmut_attn, mha_xb);
				add_input(malmut_attn, weights->lw[L].attn_output);
				add_output(malmut_attn, attn_o); // xb2

				// residual connection
				auto res_conn = std::make_shared<Operator>(OpType::OP_RES_CONN);
				g.nodes.push_back(res_conn);
				add_input(res_conn, x); // x
				add_input(res_conn, attn_o);
			}
			// ffn.build_graph(g);
			{
				auto ffn_rms_norm = std::make_shared<Operator>(OpType::OP_RMS_NORM);
				auto ffn_norm_xb  = buffer->get_new_node_dim("xb");
				auto x			  = buffer->get_new_node_dim("x");
				g.nodes.push_back(ffn_rms_norm);
				g.nodes.push_back(ffn_norm_xb);
				g.nodes.push_back(x);
				add_input(ffn_rms_norm, x);
				add_input(ffn_rms_norm, weights->lw[L].ffn_norm);
				add_output(ffn_rms_norm, ffn_norm_xb);

				auto mulmat_gate = std::make_shared<Operator>(OpType::OP_MUL_MAT);
				auto hb			 = buffer->get_new_node_hidden_dim("hb");
				g.nodes.push_back(mulmat_gate);
				g.nodes.push_back(hb);
				add_input(mulmat_gate, ffn_norm_xb);
				add_input(mulmat_gate, weights->lw[L].ffn_gate);
				add_output(mulmat_gate, hb);

				auto mulmat_up = std::make_shared<Operator>(OpType::OP_MUL_MAT);
				auto hb2	   = buffer->get_new_node_hidden_dim("hb2");
				g.nodes.push_back(mulmat_up);
				g.nodes.push_back(hb2);
				add_input(mulmat_up, ffn_norm_xb);
				add_input(mulmat_up, weights->lw[L].ffn_up);
				add_output(mulmat_up, hb2);

				auto silu = std::make_shared<Operator>(OpType::OP_SILU_HADAMARD);
				g.nodes.push_back(silu);
				add_input(silu, hb);
				add_input(silu, hb2);

				auto mulmat_down = std::make_shared<Operator>(OpType::OP_MUL_MAT);
				g.nodes.push_back(mulmat_down);
				add_input(mulmat_down, hb);
				add_input(mulmat_down, weights->lw[L].ffn_down);
				add_output(mulmat_down, ffn_norm_xb);

				auto ffn_res_conn = std::make_shared<Operator>(OpType::OP_RES_CONN);
				auto x2			  = buffer->get_new_node_dim("x");
				g.nodes.push_back(ffn_res_conn);
				g.nodes.push_back(x2);
				add_input(ffn_res_conn, x2);
				add_input(ffn_res_conn, ffn_norm_xb);
			}
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