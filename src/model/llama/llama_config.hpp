#pragma once

#include "common.hpp"
#include "ggml.h"
#include "model/common/config.hpp"

namespace smart {

struct LlamaConfig : Config {
public:
    float rope_freq_scale   = 0.0f; // rope_freq_scale (inverse of the kv) is optional
    uint32_t rope_dim_count = 0;
    uint32_t n_embd         = 0;
    uint32_t n_embd_head_k  = 0; // n_embd / n_heads
    uint32_t n_embd_head_v  = 0; // n_embd / n_heads
    uint32_t n_rot          = 0; // rope_dim_count
    uint32_t n_ctx_orig     = 0;
    float yarn_ext_factor   = 0.0f; // linear scaling factor, 1.0f for yarn
    float rope_attn_factor  = 1.0f;

public:
    LlamaConfig(gguf_context *ctx) {
        tf_cfg.dim        = get_u32(ctx, "llama.embedding_length");
        tf_cfg.hidden_dim = get_u32(ctx, "llama.feed_forward_length");
        tf_cfg.n_heads    = get_u32(ctx, "llama.attention.head_count");
        tf_cfg.n_kv_heads = get_u32(ctx, "llama.attention.head_count_kv");
        tf_cfg.n_layers   = get_u32(ctx, "llama.block_count");
        tf_cfg.seq_len    = get_u32(ctx, "llama.context_length");
        tf_cfg.vocab_size = get_u32(ctx, "llama.vocab_size");

        n_embd                = tf_cfg.dim;
        tf_cfg.rope_freq_base = get_f32(ctx, "llama.rope.freq_base", false, 10000.0f);
        // TODO: non-transformer models do not have attention heads
        if (tf_cfg.n_heads > 0) {
            n_embd_head_k  = get_u32(ctx, "llama.attention.key_length", false, n_embd / tf_cfg.n_heads);
            n_embd_head_v  = get_u32(ctx, "llama.attention.value_length", false, n_embd / tf_cfg.n_heads);
            rope_dim_count = get_u32(ctx, "llama.rope.dimension_count", false, n_embd_head_k);
            n_rot          = rope_dim_count;
        }
        n_ctx_orig       = get_u32(ctx, "llama.rope.scaling.original_context_length", false, tf_cfg.seq_len);
        rope_attn_factor = get_f32(ctx, "llama.rope.scaling.attn_factor", false, 1.0f);
        {
            float ropescale         = 0.0f;
            int rope_freq_scale_idx = gguf_find_key(ctx, "llama.rope.scaling.factor");
            if (rope_freq_scale_idx != -1) {
                rope_freq_scale_idx = gguf_find_key(ctx, "llama.rope.scale_linear");
            }
            if (rope_freq_scale_idx != -1) {
                ropescale = gguf_get_val_f32(ctx, rope_freq_scale_idx);
            }
            rope_freq_scale = ropescale == 0.0f ? 1.0f : 1.0f / ropescale;
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

public:
    void debug_config_info() override {
        fmt::println(stderr, "dim             :{:6}", tf_cfg.dim);
        fmt::println(stderr, "hidden_dim      :{:6}", tf_cfg.hidden_dim);
        fmt::println(stderr, "n_heads         :{:6}", tf_cfg.n_heads);
        fmt::println(stderr, "n_kv_heads      :{:6}", tf_cfg.n_kv_heads);
        fmt::println(stderr, "n_layers        :{:6}", tf_cfg.n_layers);
        fmt::println(stderr, "seq_len         :{:6}", tf_cfg.seq_len);
        fmt::println(stderr, "vocab_size      :{:6}", tf_cfg.vocab_size);
        fmt::println(stderr, "rope_freq_base  :{:6}", tf_cfg.rope_freq_base);
        fmt::println(stderr, "rope_freq_scale :{:6}", rope_freq_scale);
        fmt::println(stderr, "rope_dim_count  :{:6}", rope_dim_count);
        fmt::println(stderr, "n_embd          :{:6}", n_embd);
        fmt::println(stderr, "n_embd_head_k   :{:6}", n_embd_head_k);
        fmt::println(stderr, "n_embd_head_v   :{:6}", n_embd_head_v);
        fmt::println(stderr, "n_rot           :{:6}", n_rot);
        fmt::println(stderr, "n_ctx_orig      :{:6}", n_ctx_orig);
        fmt::println(stderr, "yarn_ext_factor :{:6}", yarn_ext_factor);
        fmt::println(stderr, "rope_attn_factor:{:6}", rope_attn_factor);
    }
};

} // namespace smart
