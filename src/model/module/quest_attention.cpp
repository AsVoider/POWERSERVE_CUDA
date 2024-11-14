#include "quest_attention.hpp"

#include "common.hpp"
#include "core/tensor.hpp"

namespace smart {

TensorNode *QuestAttention::build(
    Graph &g, TensorNode *x, int64_t L, const std::vector<int> &pos, const CausalAttentionMask &mask
) {
    SMART_UNUSED(mask);
    auto cfg = m_config->tf_cfg;

    auto att_norm_w = g.add_tensor(m_weights->lw[L].attn_norm);
    auto att_norm_o = g.rms_norm(x, att_norm_w, m_config->tf_cfg.norm_eps);

    // Quest QKV

    auto q_w = g.add_tensor(m_weights->lw[L].attn_q);
    auto q   = g.mat_mul(att_norm_o, q_w);

    auto k_w = g.add_tensor(m_weights->lw[L].attn_k);
    auto k   = g.mat_mul(att_norm_o, k_w);

    auto v_w = g.add_tensor(m_weights->lw[L].attn_v);
    auto v   = g.mat_mul(att_norm_o, v_w);

    const size_t head_size = cfg.head_size;
    SMART_ASSERT(head_size == cfg.rope_dim_count);
    auto q_view = g.view_tensor(q, {head_size, cfg.n_heads, q->m_shape[1], q->m_shape[2]});
    auto k_view = g.view_tensor(k, {head_size, cfg.n_kv_heads, k->m_shape[1], k->m_shape[2]});

    auto rope_q = g.rope(q_view, pos, m_config->tf_cfg.rope_cfg); // (head_size, n_heads, bs, 1)
    auto rope_k = g.rope(k_view, pos, m_config->tf_cfg.rope_cfg); // (head_size, n_kv_heads, bs, 1)

    // multihead attention
    rope_q = g.view_tensor(rope_q, q->m_shape);
    rope_k = g.view_tensor(rope_k, k->m_shape);
    g.add_cache(rope_k, L, pos, 0, true);
    g.add_cache(v, L, pos, 0, false);

    TensorNode *att_scores;
    if (L < dense_layers) {
        att_scores = g.mha(rope_q, pos, L, cfg.n_heads);
    } else {
        att_scores = g.quest_attention(rope_q, pos, L, regions_[L - dense_layers], cfg.n_heads);
    }
    // if (L == config_->n_layers - 1) {
    //  auto att_scores_ = g.mha(rope_q, pos, L, cfg.n_heads);
    // 	g.cos_sim(att_scores, att_scores_);
    // }

    auto attn_output_w = g.add_tensor(m_weights->lw[L].attn_output);
    auto attn_o        = g.mat_mul(att_scores, attn_output_w);

    // residual connection
    auto res_conn = g.add(x, attn_o);
    return res_conn;
}

} // namespace smart
