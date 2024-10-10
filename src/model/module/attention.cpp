#include "attention.hpp"
#include "core/data_type.hpp"
#include "fmt/base.h"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
namespace smart {

TensorNode *Attention::build(Graph &g, TensorNode* x, int64_t L, TensorNode* pos_tensor, int32_t pos) 
{

	auto kv_dim		  = (config->dim * config->n_kv_heads) / config->n_heads;

	// key_cache = g.new_tensor(DataType::FP32, {kv_dim, config->n_kv_heads, config->seq_len});
	// val_cache = g.new_tensor(DataType::FP32, {kv_dim, config->n_kv_heads, config->seq_len});
	auto key_cache = g.add_tensor(gkey_cache);
	auto val_cache = g.add_tensor(gval_cache);

	auto att_norm_w = g.add_tensor(weights->lw[L].attn_norm);
	auto att_norm_o = g.rms_norm(x, att_norm_w);

	// QKV
	auto kc = g.add_tensor(*key_cache);
	auto vc = g.add_tensor(*val_cache);

	auto q_w = g.add_tensor(weights->lw[L].attn_q);
	auto q = g.mat_mul(att_norm_o, q_w);
	// TODO: update cache
	uint64_t loff = L * config->seq_len * kv_dim;

	auto k_w = g.add_tensor(weights->lw[L].attn_k);
	auto k = g.mat_mul(att_norm_o, k_w);

	auto v_w = g.add_tensor(weights->lw[L].attn_v);
	auto v = g.mat_mul(att_norm_o, v_w);


	// rope -> key_cache + loff -> val_cache + loff
	// rope_q shape:[dim,]; rope_k shape: [kv_dim,]
	auto [rope_q, rope_k] = g.rope(q, k, pos_tensor);

	// multihead attention
	g.copy(vc, v, loff + pos * kv_dim);
	g.copy(kc, rope_k, loff + pos * kv_dim);

	// auto att_scores = g.mha(rope_q, kc, vc, pos_tensor, L);
	auto att_scores = g.mha(rope_q, kc, vc, pos_tensor, L);

	auto attn_output_w = g.add_tensor(weights->lw[L].attn_output);
	auto attn_o = g.mat_mul(att_scores, attn_output_w);

	// residual connection
	auto res_conn = g.add(x, attn_o);
	return res_conn;
}

void Attention::update_cache(TensorNode* pos_tensor) {
	auto pos = ((int32_t *)pos_tensor->get<ggml::Buffer>().data)[0];
	auto kv_dim		  = (config->dim * config->n_kv_heads) / config->n_heads;
	
	// fmt::println(stderr, "pos: {}", pos);
	// if (pos == 3) {
	// 	for(int i = 0; i < kv_dim; i++) {
	// 		for(int j = 0; j < pos; j++) {
	// 			for(int k = 0; k < config->n_layers; k++) {
	// 				fmt::println(stderr, "kcache[{}][{}][{}] {}", i, j, k, ((float *)gkey_cache.get<ggml::Buffer>().data)[i + j * kv_dim + k * kv_dim * config->seq_len]);
	// 			}
	// 		}
	// 	}
	// 	exit(0);
	// }
}

} // namespace smart