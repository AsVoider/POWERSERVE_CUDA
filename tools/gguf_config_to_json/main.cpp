#include "CLI/CLI.hpp"
#include "common.hpp"
#include "ggml.h"
#include "nlohmann/json.hpp"

#include <string>

static uint32_t get_u32(gguf_context *ctx, const std::string &key, bool required = true, uint32_t default_value = 0) {
    int idx = gguf_find_key(ctx, key.c_str());
    if (idx == -1) {
        SMART_ASSERT(required == false);
        return default_value;
    }
    return gguf_get_val_u32(ctx, idx);
}

static float get_f32(gguf_context *ctx, const std::string &key, bool required = true, float default_value = 0.0f) {
    int idx = gguf_find_key(ctx, key.c_str());
    if (idx == -1) {
        SMART_ASSERT(required == false);
        return default_value;
    }
    return gguf_get_val_f32(ctx, idx);
}

static std::string get_str(
    gguf_context *ctx, const std::string &key, bool required = true, const std::string &default_value = ""
) {
    int idx = gguf_find_key(ctx, key.c_str());
    if (idx == -1) {
        SMART_ASSERT(required == false);
        return default_value;
    }
    return gguf_get_val_str(ctx, idx);
}

void collect_config(gguf_context *gguf_ctx, nlohmann::json &config);

int main(int argc, char *argv[]) {
    std::string file_path   = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama-2-7b.f32.gguf";
    std::string target_path = "./llama-2-7b.json";
    CLI::App app("Config Generator");

    app.add_option("--file-path", file_path)->required();
    app.add_option("--target-path", target_path)->required();

    CLI11_PARSE(app, argc, argv);

    ggml_context *ggml_ctx;
    gguf_context *gguf_ctx;

    gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx};
    gguf_ctx                = gguf_init_from_file(file_path.c_str(), params);
    SMART_ASSERT(gguf_ctx != nullptr);
    SMART_ASSERT(ggml_ctx != nullptr);

    nlohmann::json config;

    collect_config(gguf_ctx, config);

    std::ofstream ofs(target_path);
    ofs << config.dump(4);

    gguf_free(gguf_ctx);
}

void collect_config(gguf_context *ctx, nlohmann::json &config) {
    std::string model_arch = gguf_get_val_str(ctx, gguf_find_key(ctx, "general.architecture"));
    config["model_arch"]   = model_arch;
    auto get_arch_config([&model_arch](const std::string &c) { return fmt::format(fmt::runtime(c), model_arch); });

    { // embd_dim, ffn_dim, n_heads, n_kv_heads, n_layers, n_ctx
        config["embd_dim"]        = get_u32(ctx, get_arch_config("{}.embedding_length"));
        config["ffn_dim"]         = get_u32(ctx, get_arch_config("{}.feed_forward_length"));
        config["n_attn_heads"]    = get_u32(ctx, get_arch_config("{}.attention.head_count"));
        config["n_attn_kv_heads"] = get_u32(ctx, get_arch_config("{}.attention.head_count_kv"));
        config["n_layers"]        = get_u32(ctx, get_arch_config("{}.block_count"));
        config["n_ctx"]           = get_u32(ctx, get_arch_config("{}.context_length"));
    }
    { // vocab_size
        auto idx = gguf_find_key(ctx, get_arch_config("{}.vocab_size").c_str());
        if (idx != -1) {
            config["vocab_size"] = gguf_get_val_u32(ctx, idx);
        } else {
            idx = gguf_find_key(ctx, "tokenizer.ggml.tokens");
            SMART_ASSERT(idx != -1);
            config["vocab_size"] = gguf_get_arr_n(ctx, idx);
        }
    }
    { // kv_dim, norm_eps or norm_rms_eps
        if (config["n_attn_heads"] > 0) {
            auto default_kv_dim = (uint32_t)config["embd_dim"] / (uint32_t)config["n_attn_heads"];
            auto k_dim          = get_u32(ctx, get_arch_config("{}.attention.key_length"), false, default_kv_dim);
            auto v_dim          = get_u32(ctx, get_arch_config("{}.attention.value_length"), false, default_kv_dim);
            SMART_ASSERT(k_dim == v_dim);
            config["kv_dim"] = k_dim;
        } else {
            config["kv_dim"] = 0;
        }

        auto idx      = gguf_find_key(ctx, get_arch_config("{}.attention.layer_norm_epsilon").c_str());
        auto norm_eps = 1e-5f;
        if (idx == -1) {
            idx = gguf_find_key(ctx, get_arch_config("{}.attention.layer_norm_rms_epsilon").c_str());
        }
        if (idx != -1) {
            norm_eps = gguf_get_val_f32(ctx, idx);
        }
        config["norm_eps"] = norm_eps;
    }
    { // rope_dim
        config["rope_dim"] =
            get_u32(ctx, get_arch_config("{}.rope.dimension_count"), false, (uint32_t)config["kv_dim"]);
        config["rope_freq_base"]  = std::to_string(get_f32(ctx, get_arch_config("{}.rope.freq_base"), false, 10000.0f));
        auto scale_type           = get_str(ctx, get_arch_config("{}.rope.scaling.type"), false, "linear");
        config["rope_scale_type"] = scale_type;
        config["rope_attn_factor"] =
            std::to_string(get_f32(ctx, get_arch_config("{}.rope.scaling.attn_factor"), false, 1.0f));
        config["n_rope_ctx_orig"] =
            get_u32(ctx, get_arch_config("{}.rope.scaling.original_context_length"), false, (uint32_t)config["n_ctx"]);

        auto rope_scale = 0.0f;
        auto idx        = gguf_find_key(ctx, get_arch_config("{}.rope.scaling.factor").c_str());
        if (idx == -1) {
            idx = gguf_find_key(ctx, get_arch_config("{}.rope.scale_linear").c_str());
        }
        if (idx != -1) {
            rope_scale = gguf_get_val_f32(ctx, idx);
        }
        rope_scale                = rope_scale == 0.0f ? 1.0f : 1.0f / rope_scale;
        config["rope_freq_scale"] = std::to_string(rope_scale);
    }
}
