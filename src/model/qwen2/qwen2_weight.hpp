#pragma once

#include "model/common/weights.hpp"

namespace smart {

struct Qwen2LayerWeights : LayerWeights {

public:
    Qwen2LayerWeights(ggml_context *ctx, uint32_t layer) {
        attn_norm   = get_tensor(ctx, layer, "attn_norm.weight");
        ffn_norm    = get_tensor(ctx, layer, "ffn_norm.weight");
        attn_q      = get_tensor(ctx, layer, "attn_q.weight");
        attn_q_bias = get_tensor(ctx, layer, "attn_q.bias");
        attn_k      = get_tensor(ctx, layer, "attn_k.weight");
        attn_k_bias = get_tensor(ctx, layer, "attn_k.bias");
        attn_v      = get_tensor(ctx, layer, "attn_v.weight");
        attn_v_bias = get_tensor(ctx, layer, "attn_v.bias");
        attn_output = get_tensor(ctx, layer, "attn_output.weight");
        ffn_gate    = get_tensor(ctx, layer, "ffn_gate.weight");
        ffn_up      = get_tensor(ctx, layer, "ffn_up.weight");
        ffn_down    = get_tensor(ctx, layer, "ffn_down.weight");
    }

    ~Qwen2LayerWeights() override = default;
};

struct Qwen2Weight : Weight {

public:
    Qwen2Weight(ggml_context *ctx, uint32_t n_layers, bool lazy_load) : Weight(ctx, lazy_load) {
        if (!lazy_load) {
            for (size_t layer = 0; layer < n_layers; layer++) {
                lw.push_back(Qwen2LayerWeights(ctx, layer));
            }
        }
    }

    ~Qwen2Weight() override = default;
};

} // namespace smart
