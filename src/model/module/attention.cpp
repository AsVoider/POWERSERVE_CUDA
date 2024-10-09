#include "attention.hpp"
#include "graph/graph.hpp"
#include "model/llama-impl/llama_buffer.hpp"
#include "model/llama-impl/llama_config.hpp"
#include "model/llama-impl/llama_weight.hpp"
#include <cstdint>
#include <memory>
namespace smart {

void Attention::build_graph(Graph &g, std::shared_ptr<LlamaConfig> config, std::shared_ptr<LlamaWeight> weights, std::shared_ptr<LlamaBuffer> buffer, int64_t L, int64_t pos) {
	auto kv_dim		  = (config->dim * config->n_kv_heads) / config->n_heads;
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

} // namespace smart