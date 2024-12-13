#pragma once

#include "core/config.hpp"
#include "core/kv_cache.hpp"

namespace smart::ggml {

struct GGMLKV {
public:
    using KVBuffer = std::vector<std::vector<float>>;

    size_t m_kv_dim     = 0;
    size_t m_n_kv_heads = 0;
    size_t m_n_ctx      = 0;
    size_t m_n_layers   = 0;
    size_t m_head_size  = 0;
    size_t m_batch_size = 0;
    std::shared_ptr<LLMConfig> m_config;

    struct GGMLChunk {
        KVBuffer key_buffer;   // [n_layers][seq_len * kv_dim]) kv_dim == n_kv_heads * head_size
        KVBuffer value_buffer; // [n_layers][seq_len * kv_dim])
        // TODO: No use
        KVBuffer current_k;           // [n_layers][batch_size * kv_dim])
        KVBuffer current_v;           // [n_layers][batch_size * kv_dim])
        std::vector<float> attn_bias; // [batch_size * n_ctx]
    };

    GGMLChunk chunk;

    struct GGMLKVInterface {
        GGMLKV &parent;
        GGMLChunk &chunk;

        GGMLKVInterface(GGMLKV &parent, GGMLChunk &chunk) : parent(parent), chunk(chunk) {}

        // Note: get entry from temporary kv
        ALWAYS_INLINE auto get_key(KVPosition token_pos) const -> KVView {
            auto &chk       = chunk.current_k[token_pos.layer_id];
            auto buffer     = chk.data() + token_pos.index * parent.m_kv_dim + token_pos.head_id * parent.m_head_size;
            size_t n_elem   = parent.m_head_size;
            size_t n_stride = sizeof(float); // TODO: transpose will change stride

            return {
                .n_elements   = n_elem,
                .element_size = sizeof(float),
                .stride       = n_stride,
                .data         = buffer,
            };
        }

        ALWAYS_INLINE auto get_value(KVPosition token_pos) const -> KVView {
            auto &chk       = chunk.current_v[token_pos.layer_id];
            auto buffer     = chk.data() + token_pos.index * parent.m_kv_dim + token_pos.head_id * parent.m_head_size;
            size_t n_elem   = parent.m_head_size;
            size_t n_stride = sizeof(float); // TODO: transpose will change stride

            return {
                .n_elements   = n_elem,
                .element_size = sizeof(float),
                .stride       = n_stride,
                .data         = buffer,
            };
        };

        // Note: get entry from KV cache
        ALWAYS_INLINE auto key_entry(KVPosition cache_pos) const -> KVView {
            auto &chk       = chunk.key_buffer[cache_pos.layer_id];
            auto buffer     = chk.data() + cache_pos.index * parent.m_kv_dim + cache_pos.head_id * parent.m_head_size;
            size_t n_elem   = parent.m_head_size;
            size_t n_stride = sizeof(float); // TODO: transpose will change stride

            return {
                .n_elements   = n_elem,
                .element_size = sizeof(float),
                .stride       = n_stride,
                .data         = buffer,
            };
        }

        ALWAYS_INLINE auto value_entry(KVPosition cache_pos) const -> KVView {
            auto &chk       = chunk.value_buffer[cache_pos.layer_id];
            auto buffer     = chk.data() + cache_pos.index * parent.m_kv_dim + cache_pos.head_id * parent.m_head_size;
            size_t n_elem   = parent.m_head_size;
            size_t n_stride = sizeof(float); // TODO: transpose will change stride

            return {
                .n_elements   = n_elem,
                .element_size = sizeof(float),
                .stride       = n_stride,
                .data         = buffer,
            };
        }

        ALWAYS_INLINE void set_mask(size_t cache_index, bool mask) {
            for (size_t i = 0; i < parent.m_batch_size; i++) {
                auto attn_bias         = chunk.attn_bias.data() + i * parent.m_n_ctx;
                attn_bias[cache_index] = mask ? -INFINITY : 0;
            }
        }
    };

public:
    std::unique_ptr<KVCache<GGMLKVInterface>> kv_cache;

public:
    GGMLKV(const std::shared_ptr<LLMConfig> &config);

    ~GGMLKV() = default;

public:
    void reset_batch_size(const size_t &batch_size) {
        if (m_batch_size == batch_size)
            return;
        m_batch_size = batch_size;

        auto &k = chunk.current_k;
        auto &v = chunk.current_v;
        for (size_t L = 0; L < m_n_layers; L++) {
            k[L].resize(0);
            v[L].resize(0);
            k[L].reserve(m_batch_size * m_kv_dim);
            v[L].reserve(m_batch_size * m_kv_dim);
        }
        chunk.attn_bias.resize(0);
        chunk.attn_bias.reserve(m_batch_size * m_n_ctx);
    }

private:
    void prepare_model_chunk();
};

} // namespace smart::ggml
