
#include "CLI/CLI.hpp"
#include "fmt/base.h"
#include "ggml.h"
#include "model/llama/llama_model.hpp"
#include "model/module/norm_attention.hpp"
#include "model/phi3/phi3_model.hpp"
#include "sampler/greedy_sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstdlib>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
    // 0. load config
    std::string file_path       = "/shared/models/Phi-3-mini-4k-instruct/Phi-3-mini-4k-instruct-F32.gguf";
    std::string tokenizer_path  = "/shared/models/Phi-3-mini-4k-instruct/Phi-3-mini-4k-instruct-vocab.gguf";
    float temperature           = 1.0f;       // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp                  = 0.9f;       // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps                   = 16;          // number of steps to run for
    std::string prompt          = "I was a teacher"; // prompt string
    std::string attn_type       = "normal";
    unsigned long long rng_seed = 2024927;

    CLI::App app("Demo program for llama3");

    app.add_option("--file-path", file_path)->required();
    app.add_option("--vocab-path", tokenizer_path)->required();
    app.add_option("--prompt", prompt);
    app.add_option("--steps", steps);
    app.add_option("--attn-type", attn_type);
    CLI11_PARSE(app, argc, argv);

    // 0. get model type
    std::string model_arch;
    {
        ggml_context *ggml_ctx;
        gguf_context *gguf_ctx;

        gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx};
        gguf_ctx                = gguf_init_from_file(file_path.c_str(), params);
        SMART_ASSERT(gguf_ctx != nullptr);
        SMART_ASSERT(ggml_ctx != nullptr);
        model_arch = gguf_get_val_str(gguf_ctx, gguf_find_key(gguf_ctx, "general.architecture"));
        gguf_free(gguf_ctx);
    }

    std::unique_ptr<smart::Model> model;
    if(model_arch == "llama") {
        model = std::make_unique<smart::LlamaModel>(file_path);
    }
    else if(model_arch == "phi3") {
        model = std::make_unique<smart::Phi3Model>(file_path);
    }
    else {
        fmt::print("Unknown model type\n");
    }

    if(attn_type == "normal") {
        model->m_attn = std::make_shared<smart::NormAttention>(model->m_config, model->m_weights);
    }
    else if(attn_type == "quest") {
        model->m_attn = std::make_shared<smart::QuestAttention>(model->m_config, model->m_weights);
    }

    // load tokenizer
    smart::Tokenizer tokenizer(tokenizer_path);

    // load sampler
    smart::GreedySampler sampler;

    {
        fmt::println("file_path : {}", file_path);
        fmt::println("vocab_path: {}", tokenizer_path);
        fmt::println("prompt    : {}", prompt);
        fmt::println("steps     : {}", steps);
        fmt::println("attn_type : {}", attn_type);
        fmt::println("model arch: {}", model_arch);
    }

    // generate
    model->generate(&tokenizer, (smart::Sampler *)(&sampler), prompt, steps);

}
