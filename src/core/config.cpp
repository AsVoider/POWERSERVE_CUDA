#include "core/config.hpp"

#include "common/logger.hpp"
#include "common/type_def.hpp"
#include "nlohmann/json.hpp"
#include "uv.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>

namespace smart {

HyperParams::HyperParams(const Path &params_file) {
    nlohmann::json j;
    std::ifstream file(params_file);
    file >> j;

    j["n_predicts"].get_to(n_predicts);

    j["n_threads"].get_to(n_threads);
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
        sampler_j["seed"].get_to(sampler_config.seed);
        sampler_j["temperature"].get_to(sampler_config.temperature);
        sampler_j["top_p"].get_to(sampler_config.top_p);
        sampler_j["top_k"].get_to(sampler_config.top_k);
        sampler_j["min_keep"].get_to(sampler_config.min_keep);

        sampler_j["penalty_last_n"].get_to(sampler_config.penalty_last_n);
        sampler_j["penalty_repeat"].get_to(sampler_config.penalty_repeat);
        sampler_j["penalty_freq"].get_to(sampler_config.penalty_freq);
        sampler_j["penalty_present"].get_to(sampler_config.penalty_present);
        sampler_j["penalize_nl"].get_to(sampler_config.penalize_nl);
        sampler_j["ignore_eos"].get_to(sampler_config.ignore_eos);
    }
}

ModelConfig::ModelConfig(const Path &model_config_file) {
    nlohmann::json j;
    std::ifstream file(model_config_file);
    SMART_ASSERT(file.good(), "failed to open model config file: {}", model_config_file);
    file >> j;

    j["version"].get_to(version);
    j["model_arch"].get_to(arch);
    j["model_id"].get_to(model_id);
    {
        auto &llm_info = j.at("llm_config");
        llm_info["embed_dim"].get_to(llm.dim);
        llm_info["ffn_dim"].get_to(llm.hidden_dim);
        llm_info["n_layers"].get_to(llm.n_layers);
        llm_info["n_attn_heads"].get_to(llm.n_heads);
        llm_info["n_attn_kv_heads"].get_to(llm.n_kv_heads);
        llm_info["n_ctx"].get_to(llm.seq_len);
        llm_info["vocab_size"].get_to(llm.vocab_size);
        llm_info["kv_dim"].get_to(llm.kv_dim);
        llm_info["head_size"].get_to(llm.head_size);
        llm_info["norm_eps"].get_to(llm.norm_eps);
        {
            auto &rope_info = llm_info.at("rope_config");
            rope_info["rope_dim"].get_to(llm.rope_config.n_dims);
            rope_info["n_rope_ctx_orig"].get_to(llm.rope_config.n_ctx_orig);
            rope_info["rope_freq_base"].get_to(llm.rope_config.freq_base);
            rope_info["rope_freq_scale"].get_to(llm.rope_config.freq_scale);
            llm.rope_config.ext_factor = 0.0f; // TODO: depends on scale type
            rope_info["rope_attn_factor"].get_to(llm.rope_config.attn_factor);
            // TODO: config by user
            llm.rope_config.beta_fast = 32.0f;
            llm.rope_config.beta_slow = 0.0f;
            rope_info["rope_type"].get_to(llm.rope_config.rope_type);
        }
    }
    {
        if (j.contains("vision_config")) {
            auto &vision_config = j.at("vision_config");
            vision_config["embed_dim"].get_to(vision.embed_dim);
            vision_config["num_channels"].get_to(vision.in_chans);
            vision_config["image_size"].get_to(vision.image_size);
            vision_config["num_tokens_per_patch"].get_to(vision.num_tokens_per_patch);
            vision_config["num_patches"].get_to(vision.num_patches);
        }
    }
}

Config::Config(const Path &work_folder) {
    SMART_ASSERT(std::filesystem::is_directory(work_folder));
    nlohmann::json j;
    std::ifstream file(work_folder / ARTIFACT_CONFIG_FILENAME);
    file >> j;

    if (j.contains(HYPER_PARAMS_FILENAME_KEY)) {
        hyper_params = HyperParams(work_folder / j[HYPER_PARAMS_FILENAME_KEY].get<std::string>());
    } else {
        hyper_params = HyperParams();
    }

    main_model_dir = "";
    if (j.contains(MAIN_MODEL_KEY)) {
        main_model_dir = work_folder / j[MAIN_MODEL_KEY].get<std::string>();
    }

    draft_model_dir = "";
    if (j.contains(DRAFT_MODEL_KEY)) {
        draft_model_dir = work_folder / j[DRAFT_MODEL_KEY].get<std::string>();
    }
}
} // namespace smart
