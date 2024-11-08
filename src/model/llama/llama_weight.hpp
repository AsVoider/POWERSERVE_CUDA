#pragma once

#include "model/common/weights.hpp"

namespace smart {

struct LlamaLayerWeights : LayerWeights {

public:
    LlamaLayerWeights(ggml_context *ctx, uint32_t layer) {
        attn_norm   = get_tensor(ctx, layer, "attn_norm.weight");
        ffn_norm    = get_tensor(ctx, layer, "ffn_norm.weight");
        attn_q      = get_tensor(ctx, layer, "attn_q.weight");
        attn_k      = get_tensor(ctx, layer, "attn_k.weight");
        attn_v      = get_tensor(ctx, layer, "attn_v.weight");
        attn_output = get_tensor(ctx, layer, "attn_output.weight");
        ffn_gate    = get_tensor(ctx, layer, "ffn_gate.weight");
        ffn_up      = get_tensor(ctx, layer, "ffn_up.weight");
        ffn_down    = get_tensor(ctx, layer, "ffn_down.weight");
    }

    ~LlamaLayerWeights() override = default;
};

struct LlamaWeight : Weight {

public:
    LlamaWeight(ggml_context *ctx, uint32_t n_layers) : Weight(ctx) {
        for (size_t layer = 0; layer < n_layers; layer++) {
            lw.push_back(LlamaLayerWeights(ctx, layer));
        }
    }

    ~LlamaWeight() override = default;
};

} // namespace smart
