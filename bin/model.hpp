#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"

namespace smart {

struct Config {
    uint32_t dim = 0;
    uint32_t hidden_dim = 0;
    uint32_t n_layers = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;
    uint32_t vocab_size = 0;
    uint32_t seq_len = 0;
    uint32_t rope_dim_count = 0;
};

struct RunState {
    OpTensor *x;
    OpTensor *xb;
    OpTensor *xb2;
    OpTensor *hb;
    OpTensor *hb2;
    OpTensor *q;
    OpTensor *k;
    OpTensor *v;
    OpTensor *att;
    OpTensor *logits;
    // kv cache
    OpTensor *key_cache;
    OpTensor *value_cache;
};

struct LayerWeights {
    OpTensor *attn_norm;   // "blk.$.attn_norm.weight" (layer, dim)
    OpTensor *ffn_norm;    // "blk.$.ffn_norm.weight" (layer, dim)
    // dim == n_heads * head_size
    OpTensor *attn_q;      // "blk.$.attn_q.weight" (layer, dim, n_heads * head_size)
    OpTensor *attn_k;      // "blk.$.attn_k.weight" (layer, dim, n_kv_heads * head_size)
    OpTensor *attn_v;      // "blk.$.attn_v.weight" (layer, dim, n_kv_heads * head_size)
    OpTensor *attn_output; // "blk.$.attn_output.weight" (layer, n_heads * head_size, dim)
    
    OpTensor *ffn_gate; // "blk.$.ffn_gate.weight" (layer, dim, hidden_dim)
    OpTensor *ffn_up;   // "blk.$.ffn_up.weight" (layer, dim, hidden_dim)
    OpTensor *ffn_down; // "blk.$.ffn_down.weight" (layer, hidden_dim, dim)
};

struct TransformerWeights {
    OpTensor *token_embedding_table; // "token_embd.weight" (vocab_size, dim)
    OpTensor *output_weight;         // "output.weight" (vocab_size, dim)
    OpTensor *rms_final_weight;      // "output_norm.weight" (dim,)
    float * fp32_embd_table;

    std::vector<LayerWeights> lw;
};

struct Transformer {
    Config config;              // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state;             // buffers for the "wave" of activations in the forward pass

    std::string filename;  // file descriptor for memory mapping
    ssize_t file_size;     // size of the checkpoint file in bytes
    gguf_context *gguf_ctx = nullptr;
    ggml_context *ggml_ctx = nullptr;
};

OpTensor *get_optensor_from_ggml(ggml_tensor *t);


void build_transformer(Transformer *t, std::string checkpoint_path);
void free_transformer(Transformer *t);

} // namespace smart
