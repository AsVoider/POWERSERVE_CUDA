// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "model/common/weights.hpp"

namespace smart {

struct InternVLLayerWeights : LayerWeights {

public:
    InternVLLayerWeights(ggml_context *ctx, uint32_t layer) {
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

    ~InternVLLayerWeights() override = default;
};

struct InternVLWeight : Weight {

public:
    InternVLWeight(ggml_context *ctx, uint32_t n_layers, bool lazy_load) : Weight(ctx, lazy_load) {
        if (!lazy_load) {
            for (size_t layer = 0; layer < n_layers; layer++) {
                lw.push_back(InternVLLayerWeights(ctx, layer));
            }
        }
    }

    ~InternVLWeight() override = default;
};

} // namespace smart
