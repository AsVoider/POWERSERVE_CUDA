#include <fstream>
#include <cassert>

#include "model.hpp"
#include "fmt/base.h"
#include "fmt/format.h"
#include "tensor.hpp"

namespace smart {

OpTensor *get_optensor_from_ggml(ggml_tensor *t) {
    assert(t != nullptr);
    OpTensor *opt = new OpTensor({
        .data = t->data,
        .type = t->type
    });
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        opt->ne[i] = t->ne[i];
        opt->nb[i] = t->nb[i];
    }
    return opt;
}

void free_optensor_deep(OpTensor *opt) {
    if (opt->data != nullptr) {
        switch (opt->type) {
            case GGML_TYPE_F32: delete (float *)opt->data; break;
            default: break;
        }
    }
    delete opt;
}

void free_optensor(OpTensor *opt) {
    delete opt;
}

void fill_config(Transformer *t) {
    Config &c = t->config;
    gguf_context *ctx = t->gguf_ctx;
    c.dim            = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.embedding_length"));
    c.hidden_dim     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.feed_forward_length"));
    c.n_heads        = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.attention.head_count"));
    c.n_kv_heads     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.attention.head_count_kv"));
    c.n_layers       = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.block_count"));
    c.seq_len        = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.context_length"));
    c.vocab_size     = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.vocab_size"));
    c.rope_dim_count = gguf_get_val_u32(ctx, gguf_find_key(ctx, "llama.rope.dimension_count"));
}

void prepare_state(Transformer *t) {
    auto &p = t->config;
    auto &state = t->state;
    auto kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    auto dim = p.dim;
    auto hidden_dim = p.hidden_dim;
    auto n_layers = p.n_layers;

    // alloc Optensors' buffer
    float *x           = new float[dim];
    float *xb          = new float[dim];
    float *xb2         = new float[dim];
    float *hb          = new float[hidden_dim];
    float *hb2         = new float[hidden_dim];
    float *q           = new float[dim];
    float *key_cache   = new float[n_layers * p.seq_len * kv_dim];
    float *value_cache = new float[n_layers * p.seq_len * kv_dim];
    float *att         = new float[p.n_heads * p.seq_len];
    float *logits      = new float[p.vocab_size];

    // ensure all mallocs went fine
    if (!x || !xb || !xb2 || !hb || !hb2 || !q || !key_cache || !value_cache || !att || !logits) {
        fmt::println(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    
    // alloc OpTensors
    {
        state.x = new OpTensor{.data = x, .type = GGML_TYPE_F32,
            .ne = {dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*dim, sizeof(float)*dim, sizeof(float)*dim}
        };
        state.xb = new OpTensor{.data = xb, .type = GGML_TYPE_F32,
            .ne = {dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*dim, sizeof(float)*dim, sizeof(float)*dim}
        };
        state.xb2 = new OpTensor{.data = xb2, .type = GGML_TYPE_F32,
            .ne = {dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*dim, sizeof(float)*dim, sizeof(float)*dim}
        };
        state.hb = new OpTensor{.data = hb, .type = GGML_TYPE_F32,
            .ne = {hidden_dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*hidden_dim, sizeof(float)*hidden_dim, sizeof(float)*hidden_dim}
        };
        state.hb2 = new OpTensor{.data = hb2, .type = GGML_TYPE_F32,
            .ne = {hidden_dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*hidden_dim, sizeof(float)*hidden_dim, sizeof(float)*hidden_dim}
        };
        state.q = new OpTensor{.data = q, .type = GGML_TYPE_F32,
            .ne = {dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*dim, sizeof(float)*dim, sizeof(float)*dim}
        };
        state.key_cache = new OpTensor{.data = key_cache, .type = GGML_TYPE_F32,
            .ne = {kv_dim, p.seq_len, n_layers, 1},
            .nb = {sizeof(float), sizeof(float)*kv_dim, sizeof(float)*kv_dim*p.seq_len, sizeof(float)*kv_dim*p.seq_len*n_layers}
        };
        state.value_cache = new OpTensor{.data = value_cache, .type = GGML_TYPE_F32,
            .ne = {kv_dim, p.seq_len, n_layers, 1},
            .nb = {sizeof(float), sizeof(float)*kv_dim, sizeof(float)*kv_dim*p.seq_len, sizeof(float)*kv_dim*p.seq_len*n_layers}
        };
        state.att = new OpTensor{.data = att, .type = GGML_TYPE_F32,
            .ne = {p.seq_len, p.n_heads, 1,1},
            .nb = {sizeof(float), sizeof(float)*p.seq_len, sizeof(float)*p.n_heads*p.seq_len, sizeof(float)*p.n_heads*p.seq_len}
        };
        state.logits = new OpTensor{.data = logits, .type = GGML_TYPE_F32,
            .ne = {p.vocab_size, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*p.vocab_size, sizeof(float)*p.vocab_size, sizeof(float)*p.vocab_size}
        };
        // variable pointer, no data buffer
        state.k = new OpTensor{.data = nullptr, .type = GGML_TYPE_F32,
            .ne = {kv_dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*kv_dim, sizeof(float)*kv_dim, sizeof(float)*kv_dim}
        };
        state.v = new OpTensor{.data = nullptr, .type = GGML_TYPE_F32,
            .ne = {kv_dim, 1, 1, 1},
            .nb = {sizeof(float), sizeof(float)*kv_dim, sizeof(float)*kv_dim, sizeof(float)*kv_dim}
        };

        // ensure all mallocs went fine
        if (!state.x || !state.xb || !state.xb2 || !state.hb || !state.hb2 || 
            !state.q || !state.k || !state.v || !state.key_cache || !state.value_cache || 
            !state.att || !state.logits) {
            fmt::println(stderr, "malloc optensors failed!\n");
            exit(EXIT_FAILURE);
        }

    }

}

void free_state(Transformer *t) {
    auto s = t->state;
    free_optensor_deep(s.x);
    free_optensor_deep(s.xb);
    free_optensor_deep(s.xb2);
    free_optensor_deep(s.hb);
    free_optensor_deep(s.hb2);
    free_optensor_deep(s.q);
    free_optensor_deep(s.k);
    free_optensor_deep(s.v);
    free_optensor_deep(s.att);
    free_optensor_deep(s.logits);
    free_optensor_deep(s.key_cache);
    free_optensor_deep(s.value_cache);
}

void prepare_weights(Transformer *t) {
    auto &w = t->weights;
    auto ctx = t->ggml_ctx;

    w.token_embedding_table = get_optensor_from_ggml(ggml_get_tensor(ctx, "token_embd.weight"));
    w.output_weight         = get_optensor_from_ggml(ggml_get_tensor(ctx, "output.weight"));
    w.rms_final_weight      = get_optensor_from_ggml(ggml_get_tensor(ctx, "output_norm.weight"));
    w.fp32_embd_table       = new float[ggml_nelements(ggml_get_tensor(ctx, "token_embd.weight"))];

    // dequantize_row_q8_0((block_q8_0 *)w.token_embedding_table->data, w.fp32_embd_table, ggml_nelements(ggml_get_tensor(ctx, "token_embd.weight")));
    dequantize_row_q4_0((block_q4_0 *)w.token_embedding_table->data, w.fp32_embd_table, ggml_nelements(ggml_get_tensor(ctx, "token_embd.weight")));

    w.lw.resize(t->config.n_layers);
    for (int layer = 0; layer < t->config.n_layers; layer++) {
        w.lw[layer].attn_norm   = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_norm.weight", layer).c_str()));
        w.lw[layer].ffn_norm    = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.ffn_norm.weight", layer).c_str()));
        w.lw[layer].attn_q      = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_q.weight", layer).c_str()));
        w.lw[layer].attn_k      = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_k.weight", layer).c_str()));
        w.lw[layer].attn_v      = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_v.weight", layer).c_str()));
        w.lw[layer].attn_output = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.attn_output.weight", layer).c_str()));
        w.lw[layer].ffn_gate    = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.ffn_gate.weight", layer).c_str()));
        w.lw[layer].ffn_up      = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.ffn_up.weight", layer).c_str()));
        w.lw[layer].ffn_down    = get_optensor_from_ggml(ggml_get_tensor(ctx, fmt::format("blk.{}.ffn_down.weight", layer).c_str()));
    }
}

void free_weights(Transformer *t) {
    auto &w = t->weights;

    free_optensor(w.token_embedding_table);
    free_optensor(w.output_weight);
    free_optensor(w.rms_final_weight);
    delete w.fp32_embd_table;

    for (int layer = 0; layer < t->config.n_layers; layer++) {
        free_optensor(w.lw[layer].attn_norm);
        free_optensor(w.lw[layer].ffn_norm);
        free_optensor(w.lw[layer].attn_q);
        free_optensor(w.lw[layer].attn_k);
        free_optensor(w.lw[layer].attn_v);
        free_optensor(w.lw[layer].attn_output);
        free_optensor(w.lw[layer].ffn_gate);
        free_optensor(w.lw[layer].ffn_up);
        free_optensor(w.lw[layer].ffn_down);
    }
    w.lw.clear();
}

void build_transformer(Transformer *t, std::string checkpoint_path) {
    // 1. get file size
    {
        std::ifstream file(checkpoint_path, std::ios::binary | std::ios::ate);
        assert(file.is_open());
        t->file_size = file.tellg();
        file.close();
    }
    // 2. load file meta data
    {
        t->filename = checkpoint_path;
        gguf_init_params params = {
            .no_alloc = false,
            .ctx = &t->ggml_ctx
        };
        t->gguf_ctx = gguf_init_from_file(t->filename.c_str(), params);
        assert(t->gguf_ctx != nullptr);
        assert(t->ggml_ctx != nullptr);
    }
    // 3. prepare data
    {
        fill_config(t);
        prepare_state(t);
        prepare_weights(t);
    }
}

void free_transformer(Transformer *t) {
    free_weights(t);
    free_state(t);
}

} // namespace smart