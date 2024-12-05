#include "ggml_kv_cache.hpp"

namespace smart::ggml {

GGMLKV::GGMLKV(const std::shared_ptr<Config> &config) :
    m_kv_dim(config->tf_cfg.kv_dim),
    m_n_kv_heads(config->tf_cfg.n_kv_heads),
    m_n_ctx(config->tf_cfg.seq_len),
    m_n_layers(config->tf_cfg.n_layers),
    m_head_size(config->tf_cfg.head_size),
    m_batch_size(1), // FIXME:
    m_config(config) {

    prepare_model_chunk();

    kv_cache = std::make_unique<KVCache<GGMLKVInterface>>(m_n_layers, m_n_kv_heads, m_n_ctx, *this, chunk);
}

void GGMLKV::prepare_model_chunk() {
    auto &key_buffer   = chunk.key_buffer;
    auto &value_buffer = chunk.value_buffer;
    auto &k            = chunk.current_k;
    auto &v            = chunk.current_v;

    key_buffer.resize(m_n_layers);
    value_buffer.resize(m_n_layers);
    size_t layer_size = m_kv_dim * m_n_ctx;
    for (size_t L = 0; L < m_n_layers; L++) {
        key_buffer[L].reserve(layer_size);
        value_buffer[L].reserve(layer_size);
    }

    k.resize(m_n_layers);
    v.resize(m_n_layers);
    for (size_t L = 0; L < m_n_layers; L++) {
        k[L].reserve(m_batch_size * m_kv_dim);
        v[L].reserve(m_batch_size * m_kv_dim);
    }

    auto &attn_bias = chunk.attn_bias;
    attn_bias.reserve(m_batch_size * m_n_ctx);
}

} // namespace smart::ggml
