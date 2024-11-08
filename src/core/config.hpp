#pragma once

#include "common.hpp"

#include <cstdint>

namespace smart {

struct RopeConfig {
    int n_dims        = 128; // rope_dim_count
    int n_ctx_orig    = 2048;
    float freq_base   = 10000.0f;
    float freq_scale  = 1.0f;
    float ext_factor  = 0.0f; // linear scaling factor, 1.0f for yarn
    float attn_factor = 1.0f;
    float beta_fast   = 32.0f;
    float beta_slow   = 0.0f;

public:
    void debug_config_info() {
        fmt::println(stderr, "n_dims          :{:6}", n_dims);
        fmt::println(stderr, "n_ctx_orig      :{:6}", n_ctx_orig);
        fmt::println(stderr, "freq_base       :{:6}", freq_base);
        fmt::println(stderr, "freq_scale      :{:6}", freq_scale);
        fmt::println(stderr, "ext_factor      :{:6}", ext_factor);
        fmt::println(stderr, "attn_factor     :{:6}", attn_factor);
        fmt::println(stderr, "beta_fast       :{:6}", beta_fast);
        fmt::println(stderr, "beta_slow       :{:6}", beta_slow);
    }
};

struct TransformerConfig {
public:
    uint32_t dim        = 0; // n_embd
    uint32_t hidden_dim = 0;
    uint32_t n_layers   = 0;
    uint32_t n_heads    = 0;
    uint32_t n_kv_heads = 0;
    uint32_t seq_len    = 0; // n_ctx_orig in rope
    RopeConfig rope_cfg;

    // optional
    uint32_t vocab_size     = 0;
    uint32_t rope_dim_count = 0; // n_rot in rope
    float rope_freq_base    = 10000.0f;
    uint32_t n_embd_head_k  = 0; // n_embd / n_heads
    uint32_t n_embd_head_v  = 0; // n_embd / n_heads

public:
    TransformerConfig()          = default;
    virtual ~TransformerConfig() = default;

public:
    void debug_config_info() {
        fmt::println(stderr, "dim             :{:6}", dim);
        fmt::println(stderr, "hidden_dim      :{:6}", hidden_dim);
        fmt::println(stderr, "n_layers        :{:6}", n_layers);
        fmt::println(stderr, "n_heads         :{:6}", n_heads);
        fmt::println(stderr, "n_kv_heads      :{:6}", n_kv_heads);
        fmt::println(stderr, "seq_len         :{:6}", seq_len);
        fmt::println(stderr, "vocab_size      :{:6}", vocab_size);
        fmt::println(stderr, "rope_dim_count  :{:6}", rope_dim_count);
        fmt::println(stderr, "rope_freq_base  :{:6}", rope_freq_base);
        fmt::println(stderr, "n_embd_head_k   :{:6}", n_embd_head_k);
        fmt::println(stderr, "n_embd_head_v   :{:6}", n_embd_head_v);

        rope_cfg.debug_config_info();
    }
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
    void debug_config_info() {
        tf_cfg.debug_config_info();
    }
};

} // namespace smart
