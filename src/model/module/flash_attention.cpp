#include "flash_attention.hpp"

namespace powerserve {

TensorNode *FlashAttention::build(
    Graph &g,
    TensorNode *x,
    int64_t L,
    const TensorNode *k_cache,
    const TensorNode *v_cache,
    const std::vector<int> &pos,
    const CausalAttentionMask &mask,
    bool is_need_bias = false
) {
    auto batch_size{pos.size()};
    auto head_size{m_config.head_size};
    POWERSERVE_ASSERT(head_size == (size_t)m_config.rope_config.n_dims);
    auto n_head    = m_config.n_heads;
    auto n_head_kv = m_config.n_kv_heads;
    auto n_ctx     = m_config.seq_len;

    { n_ctx = n_ctx > 1024 ? 1024 : n_ctx; }

    size_t kv_gqa  = head_size * n_head_kv;
    size_t cur_pos = pos[0];

    auto att_norm_w    = g.add_tensor(m_weights->lw[L].attn_norm);     // (embd_dim, 1, 1, 1)
    auto att_norm_o    = g.rms_norm(x, att_norm_w, m_config.norm_eps); // (embd_dim, bs, 1, 1)
    att_norm_o->m_name = fmt::format("attn_norm_o_{}", L);

    // QKV
    auto q_w  = g.add_tensor(m_weights->lw[L].attn_q); // (embd_dim, embd_dim, 1, 1)
    auto q    = g.mat_mul(q_w, att_norm_o);            // (embd_dim, bs, 1, 1)
    q->m_name = fmt::format("q_{}_{}", L, pos[0]);
    if (is_need_bias) {
        auto q_b = g.add_tensor(m_weights->lw[L].attn_q_bias); // (embd_dim, 1, 1, 1)
        q        = g.add(q, q_b);
    }
    // embd_dim == n_heads * head_size
    // kv_dim == n_kv_heads * head_size
    auto k_w  = g.add_tensor(m_weights->lw[L].attn_k); // (embd_dim, kv_dim, 1, 1)
    auto k    = g.mat_mul(k_w, att_norm_o);            // (kv_dim, batch_size, 1, 1)
    k->m_name = fmt::format("k_{}_{}", L, pos[0]);
    if (is_need_bias) {
        auto k_b = g.add_tensor(m_weights->lw[L].attn_k_bias); // (kv_dim, 1, 1, 1)
        k        = g.add(k, k_b);
    }

    auto v_w  = g.add_tensor(m_weights->lw[L].attn_v); // (embd_dim, kv_dim, 1, 1)
    auto v    = g.mat_mul(v_w, att_norm_o);            // (kv_dim, batch_size, 1, 1)
    v->m_name = fmt::format("v_{}_{}", L, pos[0]);
    if (is_need_bias) {
        auto v_b = g.add_tensor(m_weights->lw[L].attn_v_bias); // (kv_dim, 1, 1, 1)
        v        = g.add(v, v_b);
    }

    // (head_size, n_heads, bs, 1)
    auto q_view    = g.view_tensor(q, {head_size, n_head, q->m_shape[1], q->m_shape[2]});
    q_view->m_name = fmt::format("q_view_{}_{}", L, pos[0]);
    // (head_size, n_kv_heads, bs, 1)
    auto k_view    = g.view_tensor(k, {head_size, n_head_kv, k->m_shape[1], k->m_shape[2]});
    k_view->m_name = fmt::format("k_view_{}_{}", L, pos[0]);

    auto rope_factor = m_weights->rope_freq_weight.m_backend != TensorBackend::UNKNOWN
                           ? g.add_tensor(m_weights->rope_freq_weight)
                           : nullptr;
    auto rope_q      = g.rope(q_view, rope_factor, pos, m_config.rope_config); // (head_size, n_heads, bs, 1)
    rope_q->m_name   = fmt::format("rope_q_{}_{}", L, pos[0]);
    auto rope_k      = g.rope(k_view, rope_factor, pos, m_config.rope_config); // (head_size, n_kv_heads, bs, 1)
    rope_k->m_name   = fmt::format("rope_k_{}_{}", L, pos[0]);

    // store kv
    {
        k = rope_k;
        auto k_cache_view{g.view(
            k_cache,
            {batch_size * kv_gqa, 1, 1, 1},
            {k_cache->element_size(),
             k_cache->element_size() * batch_size * kv_gqa,
             k_cache->element_size() * batch_size * kv_gqa,
             k_cache->element_size() * batch_size * kv_gqa},
            k_cache->row_size(kv_gqa) * cur_pos
        )};
        k_cache_view->m_name = fmt::format("k_cache_view_{}_{}", L, pos[0]);

        auto v_cache_view{g.view(
            v_cache,
            {batch_size * kv_gqa, 1, 1, 1},
            {v_cache->element_size(),
             v_cache->element_size() * batch_size * kv_gqa,
             v_cache->element_size() * batch_size * kv_gqa,
             v_cache->element_size() * batch_size * kv_gqa},
            v_cache->row_size(kv_gqa) * cur_pos
        )};
        v_cache_view->m_name = fmt::format("v_cache_view_{}_{}", L, pos[0]);

        g.copy(k_cache_view, k);
        g.copy(v_cache_view, v);
    }

    // flash attention
    TensorNode *reshaped_fattn_res{nullptr};
    {
        size_t n_kv     = pos.back() + 1;
        n_kv            = (n_kv + KV_PADDING - 1) / KV_PADDING * KV_PADDING;
        size_t batch_32 = batch_size;

        q         = g.permute(rope_q, {0, 2, 1, 3});
        q->m_name = fmt::format("q_permute_{}_{}", L, pos[0]);

        k = g.view(
            k_cache,
            {head_size, n_kv, n_head_kv, 1},
            {k_cache->element_size(),
             k_cache->row_size(n_head_kv * head_size),
             k_cache->row_size(head_size),
             k_cache->row_size(n_head_kv * head_size)},
            0UL
        );
        k->m_name = fmt::format("k_view_{}_{}", L, pos[0]);

        v = g.view(
            v_cache,
            {head_size, n_kv, n_head_kv, 1},
            {v_cache->element_size(),
             v_cache->row_size(n_head_kv * head_size),
             v_cache->row_size(head_size),
             v_cache->row_size(n_head_kv * head_size)},
            0UL
        );
        v->m_name = fmt::format("v_view_{}_{}", L, pos[0]);

        auto f_attention_scale = 0.0f;
        float kq_scale         = f_attention_scale == 0.0f ? 1.0f / sqrtf(float(head_size)) : f_attention_scale;
        float f_max_alibi_bias = 0.000000;
        float logit_softcap    = 0.0f;

        auto kq_mask    = g.get_mask(mask, {head_size, n_kv, 1, 1}, pos, q);
        kq_mask->m_name = fmt::format("kq_mask_{}_{}", L, pos[0]);

        auto fattn_res    = g.flash_attention(q, k, v, kq_mask, kq_scale, f_max_alibi_bias, logit_softcap);
        fattn_res->m_name = fmt::format("fattn_res_{}_{}", L, pos[0]);

        reshaped_fattn_res         = g.view_tensor(fattn_res, {head_size * n_head, batch_size, 1, 1});
        reshaped_fattn_res->m_name = fmt::format("reshaped_fattn_res_{}_{}", L, pos[0]);
    }

    auto attn_out_w = g.add_tensor(m_weights->lw[L].attn_output);
    auto attn_o     = g.mat_mul(attn_out_w, reshaped_fattn_res);
    attn_o->m_name  = fmt::format("attn_o_{}_{}", L, pos[0]);

    auto res_conn    = g.add(x, attn_o);
    res_conn->m_name = fmt::format("res_conn_{}_{}", L, pos[0]);

    return res_conn;
}

} // namespace powerserve
