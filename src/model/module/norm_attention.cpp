#include "norm_attention.hpp"

#include "graph/graph.hpp"
#include "graph/node.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace smart {

TensorNode *NormAttention::build(
    Graph &g, TensorNode *x, int64_t L, const std::vector<int> &pos, const CausalAttentionMask &mask
) {
    SMART_UNUSED(mask);
    auto &llm_config = m_config;

    auto att_norm_w = g.add_tensor(m_weights->lw[L].attn_norm);       // (embed_dim, 1, 1, 1)
    auto att_norm_o = g.rms_norm(x, att_norm_w, llm_config.norm_eps); // (embed_dim, bs, 1, 1)

    // QKV
    auto q_w = g.add_tensor(m_weights->lw[L].attn_q); // (embed_dim, embed_dim, 1, 1)
    auto q   = g.mat_mul(att_norm_o, q_w);            // (embed_dim, bs, 1, 1)
    // embed_dim == n_heads * head_size
    // kv_dim == n_kv_heads * head_size
    auto k_w = g.add_tensor(m_weights->lw[L].attn_k); // (embed_dim, kv_dim, 1, 1)
    auto k   = g.mat_mul(att_norm_o, k_w);            // (kv_dim, batch_size, 1, 1)

    auto v_w = g.add_tensor(m_weights->lw[L].attn_v); // (embed_dim, kv_dim, 1, 1)
    auto v   = g.mat_mul(att_norm_o, v_w);            // (kv_dim, batch_size, 1, 1)

    const size_t head_size = llm_config.head_size;
    SMART_ASSERT(head_size == (size_t)llm_config.rope_config.n_dims);
    // (head_size, n_heads, bs, 1)
    auto q_view = g.view_tensor(q, {head_size, llm_config.n_heads, q->m_shape[1], q->m_shape[2]});
    // (head_size, n_kv_heads, bs, 1)
    auto k_view = g.view_tensor(k, {head_size, llm_config.n_kv_heads, k->m_shape[1], k->m_shape[2]});
    auto rope_q = g.rope(q_view, pos, llm_config.rope_config); // (head_size, n_heads, bs, 1)
    auto rope_k = g.rope(k_view, pos, llm_config.rope_config); // (head_size, n_kv_heads, bs, 1)

    // multihead attention
    rope_q = g.view_tensor(rope_q, q->m_shape); // (embed_dim, bs, 1, 1)
    rope_k = g.view_tensor(rope_k, k->m_shape); // (kv_dim, batch_size, 1, 1)
    g.add_cache(rope_k, L, pos, 0, true);
    g.add_cache(v, L, pos, 0, false);

    auto att_scores = g.mha(rope_q, pos, L, llm_config.n_heads); // (embed_dim, bs, 1, 1)

    auto attn_output_w = g.add_tensor(m_weights->lw[L].attn_output);
    auto attn_o        = g.mat_mul(att_scores, attn_output_w); // (embed_dim, bs, 1, 1)

    // // residual connection
    auto res_conn = g.add(x, attn_o);
    return res_conn;
}

} // namespace smart
