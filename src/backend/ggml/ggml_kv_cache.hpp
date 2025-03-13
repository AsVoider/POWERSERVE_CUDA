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

#include "core/config.hpp"
#include "core/kv_cache.hpp"
#include "core/tensor.hpp"

namespace powerserve::ggml {

class GGML_CPUKV {
public:
    static constexpr int KVPaddingSize{256};
    static constexpr int KVBlockSize{64};

    struct GGML_CPUCHUNK {
        uint8_t *cache_data_ptr{nullptr};
        size_t next_position{0};
        size_t valid_idx{0};
    };

    struct KVCacheShape {
        size_t kv_dim{0};     // 1024
        size_t kv_heads{0};   // 8
        size_t n_ctx{0};      // n
        size_t n_layers{0};   // 32
        size_t head_size{0};  // ? 128 ?
        size_t batch_size{0}; // always 1
        size_t kv_size{0};    //
        DataType type{DataType::UNKNOWN};
        bool flash_attn{false};

        auto get_k_size(size_t token_nums) -> size_t;
        auto get_v_size(size_t token_nums) -> size_t;
    };

public:
    const ModelConfig::LLMConfig &config;
    KVCacheShape kv_shape;

    std::vector<GGML_CPUCHUNK> k_cache;
    std::vector<GGML_CPUCHUNK> v_cache;

    GGML_CPUKV(const ModelConfig::LLMConfig &config);
    ~GGML_CPUKV() = default;

public:
    auto advanced_kv_cache_size(size_t token_nums) -> void;
    auto reset_kv_batch_size(size_t batch_size) -> void;
    auto get_cache_position() -> size_t;
    auto get_cache(size_t layer_id) -> std::pair<Tensor *, Tensor *>;
    auto clear_cache(size_t trunc_idx) -> void;
    auto append_k_cache(const Tensor *k_tensor, size_t layer_id, size_t token_nums) -> void;
    auto append_v_cache(const Tensor *v_tensor, size_t layer_id, size_t token_nums) -> void;
    auto get_k_cache_tensor(size_t layer_id) -> Tensor *;
    auto get_v_cache_tensor(size_t layer_id) -> Tensor *;

private:
    auto init_cache() -> void;
    auto get_k_cache(size_t layer_id) -> uint8_t *;
    auto get_v_cache(size_t layer_id) -> uint8_t *;
};

} // namespace powerserve::ggml
