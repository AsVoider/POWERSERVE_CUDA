#pragma once

#include "common.hpp"
#include "ggml.h"

#include <cstdint>

namespace smart {

struct TransformerConfig {
public:
    uint32_t dim        = 0;
    uint32_t hidden_dim = 0;
    uint32_t n_layers   = 0;
    uint32_t n_heads    = 0;
    uint32_t n_kv_heads = 0;
    uint32_t vocab_size = 0;
    uint32_t seq_len    = 0; // n_ctx_orig in rope
    // uint32_t rope_dim_count = 0; // n_rot in rope
    float rope_freq_base = 10000.0f;

public:
    TransformerConfig()          = default;
    virtual ~TransformerConfig() = default;

    // public:
    //     void debug_config_info() const {
    //         fmt::println("dim           :{:6}", dim);
    //         fmt::println("hidden_dim    :{:6}", hidden_dim);
    //         fmt::println("n_heads       :{:6}", n_heads);
    //         fmt::println("n_kv_heads    :{:6}", n_kv_heads);
    //         fmt::println("n_layers      :{:6}", n_layers);
    //         fmt::println("seq_len       :{:6}", seq_len);
    //         fmt::println("vocab_size    :{:6}", vocab_size);
    //         // fmt::println("rope_dim_count:{:6}", rope_dim_count);
    //     }
};

struct ViTConfig {
    // TODO:
};

struct Config {
public:
    TransformerConfig tf_cfg;
    ViTConfig vit_cfg;

public:
    Config()          = default;
    virtual ~Config() = default;

public:
    virtual void debug_config_info() {}
};

} // namespace smart
