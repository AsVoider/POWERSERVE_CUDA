#pragma once

#include "backend/ggml/ggml.hpp"

#include <cstdio>

namespace smart {

// Note: Each model can add its local tensors
struct LayerWeights {
public:
    Tensor attn_norm; // "blk.$.attn_norm.weight" (layer, dim)
    Tensor ffn_norm;  // "blk.$.ffn_norm.weight" (layer, dim)
    // dim == n_heads * head_size
    Tensor attn_q;      // "blk.$.attn_q.weight" (layer, dim, n_heads * head_size)
    Tensor attn_k;      // "blk.$.attn_k.weight" (layer, dim, n_kv_heads * head_size)
    Tensor attn_v;      // "blk.$.attn_v.weight" (layer, dim, n_kv_heads * head_size)
    Tensor attn_output; // "blk.$.attn_output.weight" (layer, n_heads * head_size, dim)

    Tensor ffn_gate; // "blk.$.ffn_gate.weight" (layer, dim, hidden_dim)
    Tensor ffn_up;   // "blk.$.ffn_up.weight" (layer, dim, hidden_dim)
    Tensor ffn_down; // "blk.$.ffn_down.weight" (layer, hidden_dim, dim)

    virtual ~LayerWeights() = default;

protected:
    static Tensor get_tensor(ggml_context *ctx, uint32_t layer, const char *name) {
        std::string tensor_name = fmt::format("blk.{}.{}", layer, name);
        ggml_tensor *t          = ggml_get_tensor(ctx, tensor_name.c_str());
        if (t == nullptr) {
            throw std::runtime_error(fmt::format("Failed to get tensor: {}", tensor_name));
        }
        return ggml::convert_from_ggml(t);
    }
};

struct Weight {
public:
    Tensor token_embedding_table; // "token_embd.weight" (vocab_size, dim)
    Tensor output_weight;         // "output.weight" (vocab_size, dim)
    Tensor rms_final_weight;      // "output_norm.weight" (dim,)

    std::vector<LayerWeights> lw;

public:
    Weight(ggml_context *ctx, bool lazy_load) {
        token_embedding_table = ggml::convert_from_ggml(ggml_get_tensor(ctx, "token_embd.weight"));
        if (!lazy_load) {
            auto ow_name     = ggml_get_tensor(ctx, "output.weight") == nullptr ? "token_embd.weight" : "output.weight";
            output_weight    = ggml::convert_from_ggml(ggml_get_tensor(ctx, ow_name));
            rms_final_weight = ggml::convert_from_ggml(ggml_get_tensor(ctx, "output_norm.weight"));
        }
    }

    virtual ~Weight() = default;
};

} // namespace smart
