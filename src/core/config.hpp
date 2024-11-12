#pragma once

#include "common.hpp"
#include "nlohmann/json.hpp"

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
};

struct ViTConfig {
    // TODO:
};

struct Config {
public:
    std::string arch;
    TransformerConfig tf_cfg;
    ViTConfig vit_cfg;

public:
    Config(const std::string &path);

    virtual ~Config() = default;
};

} // namespace smart
