#include "quest_attention.hpp"

#include "common.hpp"
#include "core/tensor.hpp"

namespace smart {

TensorNode *QuestAttention::build(Graph &g, TensorNode *x, int64_t L, TensorNode *pos_tensor, int32_t pos) {
    // auto kv_dim = (m_config->tf_cfg.dim * m_config->tf_cfg.n_kv_heads) / m_config->tf_cfg.n_heads;

    auto att_norm_w = g.add_tensor(m_weights->lw[L].attn_norm);
    auto att_norm_o = g.rms_norm(x, att_norm_w);

    // TODO: Quest QKV
    auto kc = m_kv_cache.add_key_cache_node(g);
    auto vc = m_kv_cache.add_value_cache_node(g);

    auto q_w = g.add_tensor(m_weights->lw[L].attn_q);
    auto q   = g.mat_mul(att_norm_o, q_w);
    // TODO: update cache
    // size_t loff = L * m_config->tf_cfg.seq_len * kv_dim;

    auto k_w = g.add_tensor(m_weights->lw[L].attn_k);
    auto k   = g.mat_mul(att_norm_o, k_w);

    auto v_w = g.add_tensor(m_weights->lw[L].attn_v);
    auto v   = g.mat_mul(att_norm_o, v_w);

    // rope -> key_cache + loff -> val_cache + loff
    // rope_q shape:[dim,]; rope_k shape: [kv_dim,]
    auto [rope_q, rope_k] = g.rope(q, k, pos_tensor);

    // multihead attention
    m_kv_cache.add_key_cache(g, rope_k, L, pos);
    m_kv_cache.add_value_cache(g, v, L, pos);

    TensorNode *att_scores;
    if (L < dense_layers) {
        att_scores = g.mha(rope_q, kc, vc, pos_tensor, L);
    } else {
        att_scores = g.quest_attention(rope_q, kc, vc, pos_tensor, L, regions_[L - dense_layers]);
    }
    // if (L == config_->n_layers - 1) {
    //  auto att_scores_ = g.mha(rope_q, kc, vc, pos_tensor, L);
    // 	g.cos_sim(att_scores, att_scores_);
    // }

    auto attn_output_w = g.add_tensor(m_weights->lw[L].attn_output);
    auto attn_o        = g.mat_mul(att_scores, attn_output_w);

    // residual connection
    auto res_conn = g.add(x, attn_o);
    return res_conn;
}

} // namespace smart
