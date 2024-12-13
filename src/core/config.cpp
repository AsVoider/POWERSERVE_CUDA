#include "core/config.hpp"

#include "nlohmann/json.hpp"

namespace smart {

HyperParams::HyperParams(const Path &params_file) {
    nlohmann::json j;
    std::ifstream file(params_file);
    file >> j;

    n_predicts = j["n_predicts"].get<size_t>();

    n_threads   = j["n_threads"].get<size_t>();
    auto n_cpus = uv_available_parallelism();
    n_threads   = std::min((unsigned int)n_threads, n_cpus);

    std::string prompt_file = j["prompt_file"].get<std::string>();
    if (prompt_file != "") {
        prompt = "";
        std::ifstream f(params_file.parent_path() / prompt_file);
        if (f.is_open()) {
            std::string line;
            while (std::getline(f, line)) {
                prompt += line + '\n';
            }
            f.close();
        } else {
            fmt::print(stderr, "Error: could not open file {}\n", prompt_file);
            SMART_ASSERT(f.is_open());
        }
    }

    auto sampler_j = j["sampler"];
    {
        sampler_config.seed        = sampler_j["seed"].get<uint64_t>();
        sampler_config.temperature = sampler_j["temperature"].get<float>();
        sampler_config.top_p       = sampler_j["top_p"].get<float>();
        sampler_config.top_k       = sampler_j["top_k"].get<size_t>();
        sampler_config.min_keep    = sampler_j["min_keep"].get<size_t>();

        sampler_config.penalty_last_n  = sampler_j["penalty_last_n"].get<int>();
        sampler_config.penalty_repeat  = sampler_j["penalty_repeat"].get<float>();
        sampler_config.penalty_freq    = sampler_j["penalty_freq"].get<float>();
        sampler_config.penalty_present = sampler_j["penalty_present"].get<float>();
        sampler_config.penalize_nl     = sampler_j["penalize_nl"].get<bool>();
        sampler_config.ignore_eos      = sampler_j["ignore_eos"].get<bool>();
    }
}

LLMConfig::LLMConfig(const Path &llm_config_file) {
    nlohmann::json j;
    std::ifstream file(llm_config_file);
    file >> j;

    version = j["version"].get<uint32_t>();

    arch       = j["model_arch"].get<std::string>();
    dim        = j["embd_dim"].get<uint32_t>();
    hidden_dim = j["ffn_dim"].get<uint32_t>();
    n_layers   = j["n_layers"].get<uint32_t>();
    n_heads    = j["n_attn_heads"].get<uint32_t>();
    n_kv_heads = j["n_attn_kv_heads"].get<uint32_t>();
    seq_len    = j["n_ctx"].get<uint32_t>();
    vocab_size = j["vocab_size"].get<uint32_t>();
    kv_dim     = j["kv_dim"].get<uint32_t>();
    head_size  = j["head_size"].get<uint32_t>();
    norm_eps   = j["norm_eps"].get<float>();

    {
        rope_config.n_dims      = j["rope_dim"].get<uint32_t>();
        rope_config.n_ctx_orig  = j["n_rope_ctx_orig"].get<uint32_t>();
        rope_config.freq_base   = j["rope_freq_base"].get<float>();
        rope_config.freq_scale  = j["rope_freq_scale"].get<float>();
        rope_config.ext_factor  = 0.0f; // TODO: depends on scale type
        rope_config.attn_factor = j["rope_attn_factor"].get<float>();
        // TODO: config by user
        rope_config.beta_fast = 32.0f;
        rope_config.beta_slow = 0.0f;
        rope_config.rope_type = j["rope_type"].get<int>();
    }
}

VisionConfig::VisionConfig(const Path &vision_config_file) {
    nlohmann::json j;
    std::ifstream file(vision_config_file);
    file >> j;
}

Config::Config(const Path &config_path) {
    SMART_ASSERT(std::filesystem::is_directory(config_path));
    nlohmann::json j;
    std::ifstream file(config_path / ARTIFACT_CONFIG_FILENAME);
    file >> j;

    if (j.contains(HYPER_PARAMS_FILENAME_KEY)) {
        hyper_params = HyperParams(config_path / j[HYPER_PARAMS_FILENAME_KEY].get<std::string>());
    } else {
        hyper_params = HyperParams();
    }

    main_llm_dir = "";
    if (j.contains(MAIN_LLM_KEY)) {
        main_llm_dir = config_path / j[MAIN_LLM_KEY].get<std::string>();
    }

    draft_llm_dir = "";
    if (j.contains(DRAFT_LLM_KEY)) {
        draft_llm_dir = config_path / j[DRAFT_LLM_KEY].get<std::string>();
    }

    if (j.contains(VISION_CONFIG_FILENAME_KEY)) {
        vision_config = VisionConfig(config_path / j[VISION_CONFIG_FILENAME_KEY].get<std::string>());
    } else {
        vision_config = VisionConfig();
    }
}

} // namespace smart
