#pragma once

#include "ggml.h"
#include "model/common/config.hpp"

namespace smart {

struct LlamaConfig : Config {
public:
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
    }

    ~LlamaConfig() = default;
};

} // namespace smart
