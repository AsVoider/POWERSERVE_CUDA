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
    gguf_context *ctx, const std::string &key, bool required = true, std::string default_value = ""
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
    CLI::App app("GGUF Converter");

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
    { // embd_dim, ffn_dim, n_heads, n_kv_heads, n_layers, n_ctx
        // TODO: name?
        config["embd_dim"]   = get_u32(ctx, fmt::format("{}.embedding_length", model_arch));
        config["ffn_dim"]    = get_u32(ctx, fmt::format("{}.feed_forward_length", model_arch));
        config["n_heads"]    = get_u32(ctx, fmt::format("{}.attention.head_count", model_arch));
        config["n_kv_heads"] = get_u32(ctx, fmt::format("{}.attention.head_count_kv", model_arch));
        config["n_layers"]   = get_u32(ctx, fmt::format("{}.block_count", model_arch));
        config["n_ctx"]      = get_u32(ctx, fmt::format("{}.context_length", model_arch));
    }
    { // vocab_size
        auto idx = gguf_find_key(ctx, fmt::format("{}.vocab_size", model_arch).c_str());
        if (idx != -1) {
            config["vocab_size"] = gguf_get_val_u32(ctx, idx);
        } else {
            idx = gguf_find_key(ctx, "tokenizer.ggml.tokens");
            SMART_ASSERT(idx != -1);
            config["vocab_size"] = gguf_get_arr_n(ctx, idx);
        }
    }
}