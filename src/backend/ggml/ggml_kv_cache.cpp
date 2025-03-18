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

#include "backend/ggml/ggml_kv_cache.hpp"

#include "backend/common/common.hpp"
#include "cpu_buffer.hpp"

namespace powerserve::ggml {

GGML_CPUKV::GGML_CPUKV(const ModelConfig::LLMConfig &config) : config{config} {
    kv_shape.kv_dim     = config.kv_dim;
    kv_shape.kv_heads   = config.n_kv_heads;
    kv_shape.n_ctx      = config.seq_len > 1024 ? 1024 : config.seq_len;
    kv_shape.n_layers   = config.n_layers;
    kv_shape.head_size  = config.head_size;
    kv_shape.batch_size = 1UL;
    kv_shape.type       = DataType::FP16;

    init_cache();
}

auto GGML_CPUKV::advanced_kv_cache_size(size_t token_nums) -> void {
    kv_shape.kv_size += token_nums;
}

auto GGML_CPUKV::reset_kv_batch_size(size_t batch_size) -> void {
    kv_shape.batch_size = batch_size;
}

auto GGML_CPUKV::get_cache_position() -> size_t {
    return kv_shape.kv_size;
}

auto GGML_CPUKV::get_cache(size_t layer_id) -> std::pair<Tensor *, Tensor *> {
    return {get_k_cache_tensor(layer_id), get_v_cache_tensor(layer_id)};
}

auto GGML_CPUKV::clear_cache(size_t trunc_idx) -> void {
    for (size_t i{0}; i < kv_shape.n_layers; ++i) {
        const auto k_cur_size{k_cache[i].next_position};
        const auto v_cur_size{v_cache[i].next_position};

        const auto k_aft_size{kv_shape.get_k_size(trunc_idx)};
        const auto v_aft_size{kv_shape.get_v_size(trunc_idx)};

        k_cache[i].valid_idx = trunc_idx;
        v_cache[i].valid_idx = trunc_idx;

        if (const size_t clear_k_size{k_cur_size - k_aft_size}; clear_k_size > 0) {
            memset(k_cache[i].cache_data_ptr + k_aft_size, 0, clear_k_size);
        }

        if (const size_t clear_v_size{v_cur_size - v_aft_size}; clear_v_size > 0) {
            if (kv_shape.flash_attn) {
                memset(v_cache[i].cache_data_ptr + v_aft_size, 0, clear_v_size);
            } else {
                // ! just do nothing here
            }
        }

        k_cache[i].next_position = k_aft_size;
        v_cache[i].next_position = v_aft_size;
    }
}

auto GGML_CPUKV::append_k_cache(const Tensor *k_tensor, size_t layer_id, size_t token_nums) -> void {
    const size_t target_size{kv_shape.get_k_size(token_nums)};
    auto dst_ptr{reinterpret_cast<void *>(k_cache[layer_id].cache_data_ptr + k_cache[layer_id].next_position)};
    auto src_ptr{const_cast<void *>(k_tensor->m_data->m_data_device)};
    memcpy(dst_ptr, src_ptr, target_size);
    k_cache[layer_id].next_position += target_size;
    k_cache[layer_id].valid_idx += token_nums;
}

auto GGML_CPUKV::append_v_cache(const Tensor *v_tensor, size_t layer_id, size_t token_nums) -> void {
    const size_t target_size{kv_shape.get_v_size(token_nums)};
    auto dst_ptr{reinterpret_cast<void *>(v_cache[layer_id].cache_data_ptr + v_cache[layer_id].next_position)};
    auto src_ptr{const_cast<void *>(v_tensor->m_data->m_data_device)};
    memcpy(dst_ptr, src_ptr, target_size);
    v_cache[layer_id].next_position += target_size;
    v_cache[layer_id].valid_idx += token_nums;
}

auto GGML_CPUKV::get_k_cache_tensor(size_t layer_id) -> Tensor * {
    auto ggml_tp{convert_datatype_to_ggml(kv_shape.type)};
    Stride t_stride{
        get_type_size(kv_shape.type),
        ggml_row_size(ggml_tp, kv_shape.kv_dim),
        ggml_row_size(ggml_tp, kv_shape.head_size),
        ggml_row_size(ggml_tp, kv_shape.kv_size * kv_shape.kv_heads * kv_shape.head_size),
    };
    Shape t_shape{
        kv_shape.head_size,
        kv_shape.kv_size,
        kv_shape.kv_heads,
        1UL,
    };
    auto ret{new Tensor{kv_shape.type, std::move(t_shape)}};
    auto cpu_buffer_ptr{std::make_shared<CPUBuffer>(
        t_stride, static_cast<void *>(get_k_cache(layer_id)), false, k_cache[layer_id].next_position, usage::COMPUTE
    )};

    ret->m_data    = cpu_buffer_ptr;
    ret->m_backend = TensorBackend::GGML_CPU;
    return ret;
}

auto GGML_CPUKV::get_v_cache_tensor(size_t layer_id) -> Tensor * {
    Stride t_stride{
        get_type_size(kv_shape.type),
        kv_shape.n_ctx * get_type_size(kv_shape.type),
        kv_shape.n_ctx * get_type_size(kv_shape.type) * kv_shape.head_size,
        kv_shape.n_ctx * get_type_size(kv_shape.type) * kv_shape.head_size * kv_shape.kv_heads,
    };
    Shape t_shape{
        kv_shape.kv_size,
        kv_shape.head_size,
        kv_shape.kv_heads,
        1UL,
    };

    auto ret{new Tensor{kv_shape.type, std::move(t_shape)}};
    auto cpu_buffer_ptr{std::make_shared<CPUBuffer>(
        t_stride, static_cast<void *>(get_v_cache(layer_id)), false, v_cache[layer_id].next_position, usage::COMPUTE
    )};

    ret->m_data    = cpu_buffer_ptr;
    ret->m_backend = TensorBackend::GGML_CPU;
    return ret;
}

auto GGML_CPUKV::init_cache() -> void {
    k_cache.resize(kv_shape.n_layers);
    v_cache.resize(kv_shape.n_layers);

    auto k_size{kv_shape.get_k_size(kv_shape.n_ctx)};
    auto v_size{kv_shape.get_v_size(kv_shape.n_ctx)};
    printf("[CPU] k_size is %ld, nctx is %ld\n", k_size, kv_shape.n_ctx);
    for (size_t i{0}; i < kv_shape.n_layers; ++i) {
        k_cache[i].cache_data_ptr = reinterpret_cast<uint8_t *>(malloc(k_size));
        v_cache[i].cache_data_ptr = reinterpret_cast<uint8_t *>(malloc(v_size));
        printf("[CPU] layer: %ld, k_cache: %p, v_cache: %p\n", i, k_cache[i].cache_data_ptr, v_cache[i].cache_data_ptr);
    }
}

auto GGML_CPUKV::get_k_cache(size_t layer_id) -> uint8_t * {
    return k_cache[layer_id].cache_data_ptr;
}

auto GGML_CPUKV::get_v_cache(size_t layer_id) -> uint8_t * {
    return v_cache[layer_id].cache_data_ptr;
}

auto GGML_CPUKV::KVCacheShape::get_k_size(size_t token_nums) -> size_t {
    auto ggml_tp{convert_datatype_to_ggml(type)};
    return ggml_row_size(ggml_tp, head_size) * kv_heads * token_nums;
}

auto GGML_CPUKV::KVCacheShape::get_v_size(size_t token_nums) -> size_t {
    auto ggml_tp{convert_datatype_to_ggml(type)};
    return flash_attn ? ggml_row_size(ggml_tp, head_size) * kv_heads * token_nums
                      : ggml_row_size(ggml_tp, token_nums) * head_size * kv_heads;
}

// GGMLKV::GGMLKV(const ModelConfig::LLMConfig &config) :
//     m_kv_dim(config.kv_dim),
//     m_n_kv_heads(config.n_kv_heads),
//     m_n_ctx(config.seq_len),
//     m_n_layers(config.n_layers),
//     m_head_size(config.head_size),
//     m_batch_size(1), // FIXME:
//     m_config(config) {

//     prepare_model_chunk();

//     kv_cache = std::make_unique<KVCache<GGMLKVInterface>>(m_n_layers, m_n_kv_heads, m_n_ctx, *this, chunk);
// }

// void GGMLKV::prepare_model_chunk() {
//     auto &key_buffer   = chunk.key_buffer;
//     auto &value_buffer = chunk.value_buffer;
//     auto &k            = chunk.current_k;
//     auto &v            = chunk.current_v;

//     key_buffer.resize(m_n_layers);
//     value_buffer.resize(m_n_layers);
//     size_t layer_size = m_kv_dim * m_n_ctx;
//     for (size_t L = 0; L < m_n_layers; L++) {
//         key_buffer[L].reserve(layer_size);
//         value_buffer[L].reserve(layer_size);

//         chunk.key_tensors.emplace_back(Tensor(DataType::FP32, {m_n_ctx, m_kv_dim, 1, 1}));
//         chunk.value_tensors.emplace_back(Tensor(DataType::FP32, {m_n_ctx, m_kv_dim, 1, 1}));
//         chunk.key_tensors.back().m_backend   = TensorBackend::GGML_CPU;
//         chunk.value_tensors.back().m_backend = TensorBackend::GGML_CPU;
//         Stride stride                        = {
//             sizeof(float),
//             sizeof(float) * m_n_ctx,
//             sizeof(float) * m_kv_dim * m_n_ctx,
//             sizeof(float) * m_kv_dim * m_n_ctx
//         };
//         chunk.key_tensors[L].m_data   = std::make_shared<CPUBuffer>(stride, key_buffer[L].data(), false, layer_size *
//         sizeof(float), usage::COMPUTE); chunk.value_tensors[L].m_data = std::make_shared<CPUBuffer>(stride,
//         value_buffer[L].data(), false, layer_size * sizeof(float), usage::COMPUTE);
//     }

//     k.resize(m_n_layers);
//     v.resize(m_n_layers);
//     for (size_t L = 0; L < m_n_layers; L++) {
//         k[L].reserve(m_batch_size * m_kv_dim);
//         v[L].reserve(m_batch_size * m_kv_dim);
//     }

//     auto &attn_bias = chunk.attn_bias;
//     attn_bias.reserve(m_batch_size * m_n_ctx);
// }

} // namespace powerserve::ggml
