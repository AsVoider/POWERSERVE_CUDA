#pragma once

#include "fmt/ostream.h"
#include "ggml.h"
#include "model/common/config.hpp"

namespace smart {

struct LlamaConfig : Config {
public:
    float rope_freq_base   = 10000.0f;
    float rope_freq_scale  = 0.0f; // rope_freq_scale (inverse of the kv) is optional
    uint32_t n_embd        = 0;
    uint32_t n_embd_head_k = 0;
    uint32_t n_embd_head_v = 0;
    uint32_t n_rot         = 0;
    uint32_t n_ctx_orig    = 0;
    float yarn_ext_factor  = 0.0f; // linear scaling factor, 1.0f for yarn
    float rope_attn_factor = 1.0f;

public:
    LlamaConfig(gguf_context *ctx) {
        tf_cfg.dim            = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.embedding_length"));
        tf_cfg.hidden_dim     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.feed_forward_length"));
        tf_cfg.n_heads        = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.attention.head_count"));
        tf_cfg.n_kv_heads     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.attention.head_count_kv"));
        tf_cfg.n_layers       = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.block_count"));
        tf_cfg.seq_len        = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.context_length"));
        tf_cfg.vocab_size     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.vocab_size"));
        tf_cfg.rope_dim_count = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.rope.dimension_count"));

        n_embd = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.embedding_length"));

        {
            int rope_freq_base_idx = gguf_find_key(ctx, "llama.rope.freq_base");
            if (rope_freq_base_idx != -1) {
                rope_freq_base = gguf_get_val_f32(ctx, rope_freq_base_idx);
            }
        }
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
        {
            // TODO: non-transformer models do not have attention heads
            bool has_heads = tf_cfg.n_heads > 0;
            if (has_heads) {
                n_embd_head_k         = n_embd / tf_cfg.n_heads;
                int n_embd_head_k_idx = gguf_find_key(ctx, "llama.attention.key_length");
                if (n_embd_head_k_idx != -1) {
                    n_embd_head_k = gguf_get_val_u32(ctx, n_embd_head_k_idx);
                }

                n_embd_head_v         = n_embd / tf_cfg.n_heads;
                int n_embd_head_v_idx = gguf_find_key(ctx, "llama.attention.value_length");
                if (n_embd_head_v_idx != -1) {
                    n_embd_head_v = gguf_get_val_u32(ctx, n_embd_head_v_idx);
                }

                n_rot         = n_embd_head_k;
                int n_rot_idx = gguf_find_key(ctx, "llama.rope.dimension_count");
                if (n_rot_idx != -1) {
                    n_rot = gguf_get_val_u32(ctx, n_rot_idx);
                }
            }
        }
        {
            uint32_t n_ctx_orig_yarn = tf_cfg.seq_len;
            int n_ctx_orig_yarn_idx  = gguf_find_key(ctx, "llama.rope.scaling.original_context_length");
            if (n_ctx_orig_yarn_idx != -1) {
                n_ctx_orig_yarn = gguf_get_val_u32(ctx, n_ctx_orig_yarn_idx);
            }
            n_ctx_orig = n_ctx_orig_yarn;
        }
        {
            int rope_attn_factor_idx = gguf_find_key(ctx, "llama.rope.scaling.attn_factor");
            if (rope_attn_factor_idx != -1) {
                rope_attn_factor = gguf_get_val_f32(ctx, rope_attn_factor_idx);
            }
        }
    }

    ~LlamaConfig() = default;

public:
    void debug_config_info() const {
        fmt::println("dim           :{:6}", tf_cfg.dim);
        fmt::println("hidden_dim    :{:6}", tf_cfg.hidden_dim);
        fmt::println("n_heads       :{:6}", tf_cfg.n_heads);
        fmt::println("n_kv_heads    :{:6}", tf_cfg.n_kv_heads);
        fmt::println("n_layers      :{:6}", tf_cfg.n_layers);
        fmt::println("seq_len       :{:6}", tf_cfg.seq_len);
        fmt::println("vocab_size    :{:6}", tf_cfg.vocab_size);
        fmt::println("rope_dim_count:{:6}", tf_cfg.rope_dim_count);
    }
};

} // namespace smart
