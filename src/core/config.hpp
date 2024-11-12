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
    Config(std::string config_path) {
        nlohmann::json j;
        std::ifstream file(config_path);
        file >> j;

        arch = std::string(j["model_arch"]);

        {
            tf_cfg.dim        = j["embd_dim"].get<uint32_t>();
            tf_cfg.hidden_dim = j["ffn_dim"].get<uint32_t>();
            tf_cfg.n_layers   = j["n_layers"].get<uint32_t>();
            tf_cfg.n_heads    = j["n_attn_heads"].get<uint32_t>();
            tf_cfg.n_kv_heads = j["n_attn_kv_heads"].get<uint32_t>();
            tf_cfg.seq_len    = j["n_ctx"].get<uint32_t>();

            tf_cfg.vocab_size     = j["vocab_size"].get<uint32_t>();
            tf_cfg.rope_dim_count = j["rope_dim"].get<uint32_t>();
            tf_cfg.rope_freq_base = std::stof((std::string)j["rope_freq_base"]);
            tf_cfg.n_embd_head_k  = j["kv_dim"].get<uint32_t>();
            tf_cfg.n_embd_head_v  = j["kv_dim"].get<uint32_t>();

            {
                auto &rope_cfg       = tf_cfg.rope_cfg;
                rope_cfg.n_dims      = tf_cfg.rope_dim_count;
                rope_cfg.n_ctx_orig  = j["n_rope_ctx_orig"].get<uint32_t>();
                rope_cfg.freq_base   = tf_cfg.rope_freq_base;
                rope_cfg.ext_factor  = 0.0f; // TODO: depends on scale type
                rope_cfg.attn_factor = std::stof((std::string)j["rope_attn_factor"]);
                // TODO: read from command args
                rope_cfg.beta_fast = 32.0f;
                rope_cfg.beta_slow = 0.0f;
            }
        }
    }

    virtual ~Config() = default;
};

} // namespace smart
