#pragma once

#include "common.hpp"
#include "ggml.h"
#include "model/common/config.hpp"

#include <cassert>

namespace smart {

struct Phi3Config : Config {

public:
    Phi3Config(gguf_context *ctx) {
        tf_cfg.seq_len        = gguf_get_val_u32(ctx, gguf_find_key(ctx, "phi3.context_length"));
        tf_cfg.dim            = gguf_get_val_u32(ctx, gguf_find_key(ctx, "phi3.embedding_length"));
        tf_cfg.hidden_dim     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "phi3.feed_forward_length"));
        tf_cfg.n_layers       = gguf_get_val_u32(ctx, gguf_find_key(ctx, "phi3.block_count"));
        tf_cfg.n_heads        = gguf_get_val_u32(ctx, gguf_find_key(ctx, "phi3.attention.head_count"));
        tf_cfg.n_kv_heads     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "phi3.attention.head_count_kv"));
        tf_cfg.rope_dim_count = gguf_get_val_u32(ctx, gguf_find_key(ctx, "phi3.rope.dimension_count"));

        auto vocab_size = gguf_find_key(ctx, "phi3.vocab_size");
        if (vocab_size != -1) {
            tf_cfg.vocab_size = gguf_get_val_u32(ctx, vocab_size);
        } else {
            const int token_idx = gguf_find_key(ctx, "tokenizer.ggml.tokens");
            SMART_ASSERT(token_idx != -1);
            tf_cfg.vocab_size = gguf_get_arr_n(ctx, token_idx);
        }
    }

    ~Phi3Config() = default;
};

} // namespace smart
