#include "norm_attention.hpp"

#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/llama/llama_config.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace smart {

TensorNode *NormAttention::build(Graph &g, TensorNode *x, int64_t L, TensorNode *pos_tensor, int32_t pos) {
    auto cfg = (LlamaConfig *)(m_config.get());

    auto att_norm_w = g.add_tensor(m_weights->lw[L].attn_norm);
    auto att_norm_o = g.rms_norm(x, att_norm_w);
    // if (L == 0 && pos == 3) g.print(att_norm_o, -1);

    // QKV
    auto kc = m_kv_cache.add_key_cache_node(g);
    auto vc = m_kv_cache.add_value_cache_node(g);

    auto q_w = g.add_tensor(m_weights->lw[L].attn_q);
    auto q   = g.mat_mul(att_norm_o, q_w);
    // if (L == 0 && pos == 3) g.print(q, -1);

    auto k_w = g.add_tensor(m_weights->lw[L].attn_k);
    auto k   = g.mat_mul(att_norm_o, k_w);
    // if (L == 0 && pos == 3) g.print(k, -1);

    auto v_w = g.add_tensor(m_weights->lw[L].attn_v);
    auto v   = g.mat_mul(att_norm_o, v_w);
    // if (L == 0 && pos == 3) g.print(v, -1);
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
    // if (L == 0 && pos == 3) g.print(rope_q, -1);
    // if (L == 0 && pos == 3) g.print(rope_k, -1);

    // multihead attention
    rope_q = g.add_tensor_view(rope_q, q->m_shape);
    rope_k = g.add_tensor_view(rope_k, k->m_shape);
    m_kv_cache.add_key_cache(g, rope_k, L, pos);
    m_kv_cache.add_value_cache(g, v, L, pos);

    auto att_scores = g.mha(rope_q, kc, vc, pos_tensor, L);

    auto attn_output_w = g.add_tensor(m_weights->lw[L].attn_output);
    auto attn_o        = g.mat_mul(att_scores, attn_output_w);
    // if (L == 0 && pos == 3) g.print(attn_o, -1);

    // residual connection
    auto res_conn = g.add(x, attn_o);
    return res_conn;
}

} // namespace smart
