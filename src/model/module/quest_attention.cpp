#include "quest_attention.hpp"

#include "common.hpp"
#include "core/tensor.hpp"

namespace smart {

TensorNode *QuestAttention::build(Graph &g, TensorNode *x, int64_t L, TensorNode *pos_tensor, int32_t pos) {
    auto cfg      = m_config->tf_cfg;
    auto rope_cfg = m_config->rope_cfg;

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
    const size_t n_embd_head = cfg.n_embd_head_v;
    SMART_ASSERT(n_embd_head == cfg.n_embd_head_k);
    SMART_ASSERT(n_embd_head == cfg.rope_dim_count);
    auto q_view      = g.view_tensor(q, {n_embd_head, cfg.n_heads, q->m_shape[2], q->m_shape[3]});
    auto k_view      = g.view_tensor(k, {n_embd_head, cfg.n_kv_heads, k->m_shape[2], k->m_shape[3]});
    auto rope_params = RopeParams{
        .n_dims      = rope_cfg.n_dims,
        .n_ctx_orig  = rope_cfg.n_ctx_orig,
        .freq_base   = rope_cfg.freq_base,
        .freq_scale  = rope_cfg.freq_scale,
        .ext_factor  = rope_cfg.ext_factor,
        .attn_factor = rope_cfg.attn_factor,
        .beta_fast   = rope_cfg.beta_fast,
        .beta_slow   = rope_cfg.beta_slow
    };
    auto rope_q = g.rope(q_view, pos_tensor, rope_params);
    auto rope_k = g.rope(k_view, pos_tensor, rope_params);

    // multihead attention
    rope_q = g.view_tensor(rope_q, q->m_shape);
    rope_k = g.view_tensor(rope_k, k->m_shape);
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
