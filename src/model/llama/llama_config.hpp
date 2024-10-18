#pragma once

#include "fmt/ostream.h"
#include "ggml.h"

#include <cstdint>

namespace smart {

struct LlamaConfig {
public:
    uint32_t dim            = 0;
    uint32_t hidden_dim     = 0;
    uint32_t n_layers       = 0;
    uint32_t n_heads        = 0;
    uint32_t n_kv_heads     = 0;
    uint32_t vocab_size     = 0;
    uint32_t seq_len        = 0;
    uint32_t rope_dim_count = 0;

public:
    LlamaConfig(gguf_context *ctx) {
        dim        = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.embedding_length"));
        hidden_dim = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.feed_forward_length"));
        n_heads    = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.attention.head_count"));
        n_kv_heads = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.attention.head_count_kv"));
        n_layers   = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.block_count"));
        seq_len    = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.context_length"));
        // vocab_size     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.vocab_size"));
        vocab_size     = 0;
        rope_dim_count = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.rope.dimension_count"));
    }

    ~LlamaConfig() = default;

public:
    void debug_config_info() const {
        fmt::println("dim           :{:6}", dim);
        fmt::println("hidden_dim    :{:6}", hidden_dim);
        fmt::println("n_heads       :{:6}", n_heads);
        fmt::println("n_kv_heads    :{:6}", n_kv_heads);
        fmt::println("n_layers      :{:6}", n_layers);
        fmt::println("seq_len       :{:6}", seq_len);
        fmt::println("vocab_size    :{:6}", vocab_size);
        fmt::println("rope_dim_count:{:6}", rope_dim_count);
    }
};

} // namespace smart
