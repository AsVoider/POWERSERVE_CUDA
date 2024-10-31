
#include "CLI/CLI.hpp"
#include "common.hpp"
#include "ggml.h"
#include "model/llama/llama_model.hpp"
#include "model/module/norm_attention.hpp"
#include "model/module/quest_attention.hpp"
#include "model/phi3/phi3_model.hpp"
#include "sampler/greedy_sampler.hpp"
#include "sampler/topp_sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstdlib>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
    // 0. load config
    std::string file_path       = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama-2-7b.f32.gguf";
    std::string tokenizer_path  = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama2_7b_vocab.gguf";
    float temperature           = 0.6f;       // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp                  = 0.9f;       // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps                   = 64;          // number of steps to run for
    std::string prompt          = "One day,"; // prompt string
    std::string attn_type       = "normal";
    int n_threads               = 4;
    uint64_t rng_seed = 2024927;

    CLI::App app("Demo program for llama3");

    app.add_option("--file-path", file_path)->required();
    app.add_option("--vocab-path", tokenizer_path)->required();
    app.add_option("--prompt", prompt);
    app.add_option("--steps", steps);
    app.add_option("--attn-type", attn_type);
    app.add_option("--n-threads", n_threads);
    app.add_option("--temperature", temperature);
    app.add_option("--topp", topp);
    app.add_option("--rng-seed", rng_seed);
    CLI11_PARSE(app, argc, argv);

    // get number of CPUs
    {
        auto n_cpus = uv_available_parallelism(); // Default fallback value
        fmt::println("n_cpus: {}", n_cpus);
        // NOTE: the main thread is also a worker thread, so we need to subtract 1
        SMART_ASSERT(n_cpus >= 2);
        n_threads = std::min((unsigned int)n_threads, n_cpus - 1);
    }

    // get model type
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
        model = std::make_unique<smart::LlamaModel>(file_path, n_threads);
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
    // smart::GreedySampler sampler;
    smart::ToppSampler sampler{temperature, topp, rng_seed};


    {
        fmt::println("file_path   : {}", file_path);
        fmt::println("vocab_path  : {}", tokenizer_path);
        fmt::println("prompt      : {}", prompt);
        fmt::println("steps       : {}", steps);
        fmt::println("attn_type   : {}", attn_type);
        fmt::println("model arch  : {}", model_arch);
        fmt::println("n_threads   : {}", n_threads);
        fmt::println("temperature : {}", temperature);
        fmt::println("topp        : {}", topp);
        fmt::println("rng_seed    : {}", rng_seed);
    }

    // generate
    model->generate(&tokenizer, (smart::Sampler *)(&sampler), prompt, steps);

}
