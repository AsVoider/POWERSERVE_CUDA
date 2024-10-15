#pragma once

#include "ggml.h"

#include <cstdint>

namespace smart {

struct LlamaConfig {
    uint32_t dim            = 0;
    uint32_t hidden_dim     = 0;
    uint32_t n_layers       = 0;
    uint32_t n_heads        = 0;
    uint32_t n_kv_heads     = 0;
    uint32_t vocab_size     = 0;
    uint32_t seq_len        = 0;
    uint32_t rope_dim_count = 0;

    LlamaConfig(gguf_context *ctx) {
        dim            = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.embedding_length"));
        hidden_dim     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.feed_forward_length"));
        n_heads        = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.attention.head_count"));
        n_kv_heads     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.attention.head_count_kv"));
        n_layers       = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.block_count"));
        seq_len        = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.context_length"));
        vocab_size     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.vocab_size"));
        rope_dim_count = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.rope.dimension_count"));
    }

    ~LlamaConfig() = default;
};

} // namespace smart
