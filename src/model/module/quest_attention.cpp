#include "quest_attention.hpp"

#include "common.hpp"
#include "core/tensor.hpp"
#include "model/llama/llama_config.hpp"

namespace smart {

TensorNode *QuestAttention::build(Graph &g, TensorNode *x, int64_t L, TensorNode *pos_tensor, int32_t pos) {
    auto cfg = (LlamaConfig *)(m_config.get());
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
    const size_t n_embd_head = cfg->n_embd_head_v;
    SMART_ASSERT(n_embd_head == cfg->n_embd_head_k);
    SMART_ASSERT(n_embd_head == cfg->n_rot);
    auto q_view = g.add_tensor_view(q, {n_embd_head, cfg->tf_cfg.n_heads, q->m_shape[2], q->m_shape[3]});
    auto k_view = g.add_tensor_view(k, {n_embd_head, cfg->tf_cfg.n_kv_heads, k->m_shape[2], k->m_shape[3]});
    auto rope_q = g.rope(
        q_view,
        // q,
        pos_tensor,
        cfg->n_rot,
        cfg->n_ctx_orig,
        cfg->tf_cfg.rope_freq_base,
        cfg->rope_freq_scale,
        cfg->yarn_ext_factor,
        cfg->rope_attn_factor
    );
    auto rope_k = g.rope(
        k_view,
        // k,
        pos_tensor,
        cfg->n_rot,
        cfg->n_ctx_orig,
        cfg->tf_cfg.rope_freq_base,
        cfg->rope_freq_scale,
        cfg->yarn_ext_factor,
        cfg->rope_attn_factor
    );

    // multihead attention
    rope_q = g.add_tensor_view(rope_q, q->m_shape);
    rope_k = g.add_tensor_view(rope_k, k->m_shape);
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
