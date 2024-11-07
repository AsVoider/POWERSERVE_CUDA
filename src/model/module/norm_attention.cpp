#include "norm_attention.hpp"

#include "graph/graph.hpp"
#include "graph/node.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace smart {

TensorNode *NormAttention::build(Graph &g, TensorNode *x, int64_t L, TensorNode *pos_tensor, int32_t pos) {
    auto &cfg = m_config->tf_cfg;

    auto att_norm_w = g.add_tensor(m_weights->lw[L].attn_norm);
    auto att_norm_o = g.rms_norm(x, att_norm_w);

    // QKV
    auto kc = m_kv_cache.add_key_cache_node(g);
    auto vc = m_kv_cache.add_value_cache_node(g);

    auto q_w = g.add_tensor(m_weights->lw[L].attn_q);
    auto q   = g.mat_mul(att_norm_o, q_w);

    auto k_w = g.add_tensor(m_weights->lw[L].attn_k);
    auto k   = g.mat_mul(att_norm_o, k_w);

    auto v_w = g.add_tensor(m_weights->lw[L].attn_v);
    auto v   = g.mat_mul(att_norm_o, v_w);

    const size_t n_embd_head = cfg.n_embd_head_v;
    SMART_ASSERT(n_embd_head == cfg.n_embd_head_k);
    SMART_ASSERT(n_embd_head == cfg.rope_dim_count);
    auto q_view = g.view_tensor(q, {n_embd_head, cfg.n_heads, q->m_shape[2], q->m_shape[3]});
    auto k_view = g.view_tensor(k, {n_embd_head, cfg.n_kv_heads, k->m_shape[2], k->m_shape[3]});
    auto rope_q = g.rope(q_view, pos_tensor, m_config->rope_cfg);
    auto rope_k = g.rope(k_view, pos_tensor, m_config->rope_cfg);

    // multihead attention
    rope_q = g.view_tensor(rope_q, q->m_shape);
    rope_k = g.view_tensor(rope_k, k->m_shape);
    m_kv_cache.add_key_cache(g, rope_k, L, pos);
    m_kv_cache.add_value_cache(g, v, L, pos);

    auto att_scores = g.mha(rope_q, kc, vc, pos_tensor, L, cfg.n_heads);

    auto attn_output_w = g.add_tensor(m_weights->lw[L].attn_output);
    auto attn_o        = g.mat_mul(att_scores, attn_output_w);

    // residual connection
    auto res_conn = g.add(x, attn_o);
    return res_conn;
}

} // namespace smart
