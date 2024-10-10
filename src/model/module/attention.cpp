#include "attention.hpp"
#include "core/data_type.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include <cstdint>
namespace smart {

TensorNode *Attention::build(Graph &g, TensorNode* x, int64_t L, int64_t pos) 
{

	auto kv_dim		  = (config->dim * config->n_kv_heads) / config->n_heads;
	auto att_norm_w = g.add_tensor(weights->lw[L].attn_norm);
	auto att_norm_o = g.rms_norm(x, att_norm_w);

	// QKV
	auto q_w = g.add_tensor(weights->lw[L].attn_q);
	auto q = g.mat_mul(att_norm_o, q_w);

	uint64_t loff = L * config->seq_len * kv_dim;

	auto k_w = g.add_tensor(weights->lw[L].attn_k);
	auto k = g.mat_mul(att_norm_o, k_w);

	auto v_w = g.add_tensor(weights->lw[L].attn_v);
	auto v = g.mat_mul(att_norm_o, v_w);

	// rope
	auto pos_tensor = g.new_tensor(DataType::INT32, {1});
	auto [rope_q, rope_k] = g.rope(q, k, pos_tensor);

	// multihead attention
	auto key_cache = g.new_tensor(DataType::FP32, {kv_dim, config->seq_len, config->n_layers});
	auto val_cache = g.new_tensor(DataType::FP32, {kv_dim, config->seq_len, config->n_layers});
	auto att_scores = g.mha(rope_q, key_cache, val_cache, pos_tensor, L);

	auto attn_output_w = g.add_tensor(weights->lw[L].attn_output);
	auto attn_o = g.mat_mul(att_scores, attn_output_w);

	// residual connection
	auto res_conn = g.add(x, attn_o);
	return res_conn;
}

} // namespace smart