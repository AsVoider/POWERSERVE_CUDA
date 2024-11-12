#include "core/config.hpp"

namespace smart {

Config::Config(const std::string &path) {
    nlohmann::json j;
    std::ifstream file(path);
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
        tf_cfg.rope_freq_base = j["rope_freq_base"].get<float>();
        tf_cfg.n_embd_head_k  = j["kv_dim"].get<uint32_t>();
        tf_cfg.n_embd_head_v  = j["kv_dim"].get<uint32_t>();

        {
            auto &rope_cfg       = tf_cfg.rope_cfg;
            rope_cfg.n_dims      = tf_cfg.rope_dim_count;
            rope_cfg.n_ctx_orig  = j["n_rope_ctx_orig"].get<uint32_t>();
            rope_cfg.freq_base   = tf_cfg.rope_freq_base;
            rope_cfg.ext_factor  = 0.0f; // TODO: depends on scale type
            rope_cfg.attn_factor = j["rope_attn_factor"].get<float>();
            // TODO: read from command args
            rope_cfg.beta_fast = 32.0f;
            rope_cfg.beta_slow = 0.0f;
        }
    }
}

} // namespace smart
