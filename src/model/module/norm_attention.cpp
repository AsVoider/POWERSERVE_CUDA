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

    // QKV
    auto kc = m_kv_cache.add_key_cache_node(g);
    auto vc = m_kv_cache.add_value_cache_node(g);

    auto q_w = g.add_tensor(m_weights->lw[L].attn_q);
    auto q   = g.mat_mul(att_norm_o, q_w);
    // g.print(q, 10);

    auto k_w = g.add_tensor(m_weights->lw[L].attn_k);
    auto k   = g.mat_mul(att_norm_o, k_w);

    auto v_w = g.add_tensor(m_weights->lw[L].attn_v);
    auto v   = g.mat_mul(att_norm_o, v_w);

    // rope -> key_cache + loff -> val_cache + loff
    // rope_q shape:[dim,]; rope_k shape: [kv_dim,]
    // auto [rope_q, rope_k] = g.rope(q, k, pos_tensor);
    auto rope_q = g.rope(
        q,
        pos_tensor,
        cfg->n_rot,
        cfg->n_ctx_orig,
        cfg->rope_freq_base,
        cfg->rope_freq_scale,
        cfg->yarn_ext_factor,
        cfg->rope_attn_factor
    );
    auto rope_k = g.rope(
        k,
        pos_tensor,
        cfg->n_rot,
        cfg->n_ctx_orig,
        cfg->rope_freq_base,
        cfg->rope_freq_scale,
        cfg->yarn_ext_factor,
        cfg->rope_attn_factor
    );

    // multihead attention
    m_kv_cache.add_key_cache(g, rope_k, L, pos);
    m_kv_cache.add_value_cache(g, v, L, pos);

    // auto att_scores = g.mha(rope_q, kc, vc, pos_tensor, L);
    auto att_scores = g.mha(rope_q, kc, vc, pos_tensor, L);

    auto attn_output_w = g.add_tensor(m_weights->lw[L].attn_output);
    auto attn_o        = g.mat_mul(att_scores, attn_output_w);

    // residual connection
    auto res_conn = g.add(x, attn_o);
    return res_conn;
}

} // namespace smart
