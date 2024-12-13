#pragma once

#include "common.hpp"

#include <cstdint>

namespace smart {

const std::string HYPER_PARAMS_FILENAME_KEY  = "params_config";
const std::string MAIN_LLM_KEY               = "llm_main";
const std::string DRAFT_LLM_KEY              = "llm_draft";
const std::string LLM_CONFIG_FILENAME        = "llm.json";
const std::string LLM_WEIGHTS_FILENAME       = "weights.gguf";
const std::string LLM_VOCAB_FILENAME         = "vocab.gguf";
const std::string VISION_CONFIG_FILENAME_KEY = "vision_config";
const std::string ARTIFACT_CONFIG_FILENAME   = "artifact.json";

struct HyperParams {
    struct SamplerConfig {
        uint64_t seed     = 0;
        float temperature = 0.80f;
        float top_p       = 0.95f; // 1.0 = disabled
        size_t top_k      = 40;
        size_t min_keep   = 0; // 0 = disabled, otherwise samplers should return at least min_keep tokens

        int penalty_last_n    = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
        float penalty_repeat  = 1.00f; // 1.0 = disabled
        float penalty_freq    = 0.00f; // 0.0 = disabled
        float penalty_present = 0.00f; // 0.0 = disabled
        bool penalize_nl      = false; // consider newlines as a repeatable token
        bool ignore_eos       = false;
    } sampler_config;

    size_t n_predicts  = 32;
    size_t n_threads   = 4;
    std::string prompt = "One day,";

    HyperParams() = default;
    HyperParams(const Path &params_file);

    ~HyperParams() = default;
};

struct SamplerConfig {
    uint64_t seed     = 0;
    float temperature = 0.80f;
    float top_p       = 0.95f; // 1.0 = disabled
    size_t top_k      = 40;
    size_t min_keep   = 0; // 0 = disabled, otherwise samplers should return at least min_keep tokens

    int penalty_last_n    = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float penalty_repeat  = 1.00f; // 1.0 = disabled
    float penalty_freq    = 0.00f; // 0.0 = disabled
    float penalty_present = 0.00f; // 0.0 = disabled
    bool penalize_nl      = false; // consider newlines as a repeatable token
    bool ignore_eos       = false;

    SamplerConfig() = default;
    SamplerConfig(const Path &sampler_config_file);

    virtual ~SamplerConfig() = default;
};

struct LLMConfig {

    struct RopeConfig {
        int n_dims        = 128; // rope_dim_count
        int n_ctx_orig    = 2048;
        float freq_base   = 10000.0f;
        float freq_scale  = 1.0f;
        float ext_factor  = 0.0f; // linear scaling factor, 1.0f for yarn
        float attn_factor = 1.0f;
        float beta_fast   = 32.0f;
        float beta_slow   = 0.0f;
        int rope_type     = -1;
    } rope_config;

    uint32_t version;
    std::string arch;

    uint32_t dim        = 0; // n_embd
    uint32_t hidden_dim = 0;
    uint32_t n_layers   = 0;
    uint32_t n_heads    = 0;
    uint32_t n_kv_heads = 0;
    uint32_t seq_len    = 0; // n_ctx_orig in rope
    uint32_t vocab_size = 0;
    uint32_t kv_dim     = 0; // head_size * n_kv_heads
    uint32_t head_size  = 0; // dim / n_heads
    float norm_eps      = 1e-5f;

    LLMConfig() = default;
    LLMConfig(const Path &llm_config_file);

    virtual ~LLMConfig() = default;
};

struct VisionConfig {

    VisionConfig() = default;
    VisionConfig(const Path &vision_config_file);

    virtual ~VisionConfig() = default;
};

struct Config {
public:
    Path main_llm_dir;
    Path draft_llm_dir;

    HyperParams hyper_params;

    std::shared_ptr<LLMConfig> main_llm_config;
    std::shared_ptr<LLMConfig> draft_llm_config;
    VisionConfig vision_config;

public:
    Config(const Path &config_path);

    virtual ~Config() = default;
};

} // namespace smart
