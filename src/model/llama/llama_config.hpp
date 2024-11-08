#pragma once

#include "common.hpp"
#include "core/config.hpp"
#include "ggml.h"

namespace smart {

struct LlamaConfig : Config {
public:
    LlamaConfig(gguf_context *ctx) {
        tf_cfg.dim        = get_u32(ctx, "llama.embedding_length");
        tf_cfg.hidden_dim = get_u32(ctx, "llama.feed_forward_length");
        tf_cfg.n_heads    = get_u32(ctx, "llama.attention.head_count");
        tf_cfg.n_kv_heads = get_u32(ctx, "llama.attention.head_count_kv");
        tf_cfg.n_layers   = get_u32(ctx, "llama.block_count");
        tf_cfg.seq_len    = get_u32(ctx, "llama.context_length");
        tf_cfg.vocab_size = get_u32(ctx, "llama.vocab_size");

        tf_cfg.rope_freq_base = get_f32(ctx, "llama.rope.freq_base", false, 10000.0f);
        // TODO: non-transformer models do not have attention heads
        if (tf_cfg.n_heads > 0) {
            tf_cfg.n_embd_head_k  = get_u32(ctx, "llama.attention.key_length", false, tf_cfg.dim / tf_cfg.n_heads);
            tf_cfg.n_embd_head_v  = get_u32(ctx, "llama.attention.value_length", false, tf_cfg.dim / tf_cfg.n_heads);
            tf_cfg.rope_dim_count = get_u32(ctx, "llama.rope.dimension_count", false, tf_cfg.n_embd_head_k);
        }
        tf_cfg.rope_cfg.n_dims      = tf_cfg.rope_dim_count;
        tf_cfg.rope_cfg.n_ctx_orig  = get_u32(ctx, "llama.rope.scaling.original_context_length", false, tf_cfg.seq_len);
        tf_cfg.rope_cfg.attn_factor = get_f32(ctx, "llama.rope.scaling.attn_factor", false, 1.0f);
        {
            float ropescale         = 0.0f;
            int rope_freq_scale_idx = gguf_find_key(ctx, "llama.rope.scaling.factor");
            if (rope_freq_scale_idx != -1) {
                rope_freq_scale_idx = gguf_find_key(ctx, "llama.rope.scale_linear");
            }
            if (rope_freq_scale_idx != -1) {
                ropescale = gguf_get_val_f32(ctx, rope_freq_scale_idx);
            }
            tf_cfg.rope_cfg.freq_scale = ropescale == 0.0f ? 1.0f : 1.0f / ropescale;
        }
    }

    ~LlamaConfig() = default;

private:
    uint32_t get_u32(gguf_context *ctx, const std::string &key, bool required = true, uint32_t default_value = 0) {
        int idx = gguf_find_key(ctx, key.c_str());
        if (idx == -1) {
            SMART_ASSERT(required == false);
            return default_value;
        }
        return gguf_get_val_u32(ctx, idx);
    }

    float get_f32(gguf_context *ctx, const std::string &key, bool required = true, float default_value = 0.0f) {
        int idx = gguf_find_key(ctx, key.c_str());
        if (idx == -1) {
            SMART_ASSERT(required == false);
            return default_value;
        }
        return gguf_get_val_f32(ctx, idx);
    }
};

} // namespace smart
