
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <cassert>
#include <fstream>

#include "ggml.h"
#include "fmt/core.h"
#include "tensor.hpp"
#include "llama_tokenizer.hpp"
#include "sampler.hpp"
#include "debug.hpp"

namespace smart {

long time_in_ms();
void safe_printf(std::string piece);

struct Transformer {

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

    Config config;              // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state;             // buffers for the "wave" of activations in the forward pass

    std::string filename;  // file descriptor for memory mapping
    ssize_t file_size;     // size of the checkpoint file in bytes
    gguf_context *gguf_ctx = nullptr;
    ggml_context *ggml_ctx = nullptr;

    Transformer(std::string checkpoint_path);
    ~Transformer();

    void generate(LlamaTokenizer *tk, Sampler *sampler, std::string prompt, int steps);
    void debug_config_info();
    void debug_weights_info();

private:
    op_compute_params params; // temp buffer for compute

    void fill_config();
    void prepare_state();
    void free_state();
    void prepare_weights();
    void free_weights();

    void multihead_attention(
        OpTensor* q_tensor, 
        OpTensor* k_cache_tensor, 
        OpTensor* attn_tensor, 
        OpTensor* v_cache_tensor, 
        OpTensor* xb_tensor, 
        uint32_t pos, 
        uint64_t loff
    );
    float* forward(int token, int pos);
};

} // namespace smart