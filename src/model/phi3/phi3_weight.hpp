#pragma once

#include "core/buffer.hpp"
#include "core/tensor.hpp"
#include "model/common/weights.hpp"

#include <cstdio>

namespace smart {

struct Phi3LayerWeights : LayerWeights {

public:
    Phi3LayerWeights(ggml_context *ctx, uint32_t layer) {
        attn_norm   = get_tensor(ctx, layer, "attn_norm.weight");
        ffn_norm    = get_tensor(ctx, layer, "ffn_norm.weight");
        attn_output = get_tensor(ctx, layer, "attn_output.weight");
        ffn_down    = get_tensor(ctx, layer, "ffn_down.weight");

        Tensor attn_qkv             = get_tensor(ctx, layer, "attn_qkv.weight");
        Tensor::Shape shape         = attn_qkv.m_shape;
        ggml::Buffer::Stride stride = attn_qkv.get<ggml::Buffer>().m_stride;
        // NOTE: phi's qkv store together
        shape[1] /= 3;
        stride[2] /= 3;
        stride[3] /= 3;

        attn_q = Tensor(attn_qkv.m_dtype, shape);
        attn_k = Tensor(attn_qkv.m_dtype, shape);
        attn_v = Tensor(attn_qkv.m_dtype, shape);

        attn_q.m_data = std::make_shared<ggml::Buffer>(stride, attn_qkv.get<ggml::Buffer>().m_data);

        void *k_ptr = (char *)(attn_qkv.get<ggml::Buffer>().m_data) + stride[1] * shape[1];
        void *v_ptr = (char *)(attn_qkv.get<ggml::Buffer>().m_data) + 2 * stride[1] * shape[1];

        attn_k.m_data = std::make_shared<ggml::Buffer>(stride, k_ptr);
        attn_v.m_data = std::make_shared<ggml::Buffer>(stride, v_ptr);

        Tensor attn_gateup = get_tensor(ctx, layer, "ffn_up.weight");
        shape              = attn_gateup.m_shape;
        stride             = attn_gateup.get<ggml::Buffer>().m_stride;
        // NOTE: phi's gate and up store together
        shape[1] /= 2;
        stride[2] /= 2;
        stride[3] /= 2;

        ffn_gate = Tensor(attn_gateup.m_dtype, shape);
        ffn_up   = Tensor(attn_gateup.m_dtype, shape);

        ffn_gate.m_data = std::make_shared<ggml::Buffer>(stride, attn_gateup.get<ggml::Buffer>().m_data);
        ffn_up.m_data   = std::make_shared<ggml::Buffer>(
            stride, (char *)(attn_gateup.get<ggml::Buffer>().m_data) + stride[1] * shape[1]
        );
    }

    ~Phi3LayerWeights() = default;
};

struct Phi3Weight : Weight {

public:
    Phi3Weight(ggml_context *ctx, uint32_t n_layers, uint32_t dim) : Weight(ctx, dim) {
        for (size_t layer = 0; layer < n_layers; layer++) {
            lw.push_back(Phi3LayerWeights(ctx, layer));
        }
    }

    ~Phi3Weight() override = default;
};

} // namespace smart
