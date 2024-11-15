#include "CLI/CLI.hpp"
#include "common.hpp"
#include "model/llama/llama_model.hpp"
#include "model/module/norm_attention.hpp"
#include "model/module/quest_attention.hpp"
#include "sampler/sampler_chain.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstdlib>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
    // 0. load config
    std::string file_path      = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama-2-7b.f32.gguf";
    std::string tokenizer_path = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama2_7b_vocab.gguf";
    std::string config_path    = "/home/zwb/SS/smartserving/llama3.2.json";
    float temperature          = 0.8f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float top_p                = 0.95f;
    size_t top_k               = 40;
    int steps                  = 64;         // number of steps to run for
    std::string prompt         = "One day,"; // prompt string
    std::string attn_type      = "normal";
    int n_threads              = 4;
    uint64_t rng_seed          = uint64_t(-1); // uint64_t(-1) = random seed

    CLI::App app("Demo program for llama3");

    app.add_option("--file-path", file_path)->required();
    app.add_option("--vocab-path", tokenizer_path)->required();
    app.add_option("--config-path", config_path)->required();
    app.add_option("--prompt", prompt);
    app.add_option("--steps", steps);
    app.add_option("--attn-type", attn_type);
    app.add_option("--n-threads", n_threads);
    app.add_option("--temperature", temperature);
    app.add_option("--top-p", top_p);
    app.add_option("--top-k", top_k);
    app.add_option("--rng-seed", rng_seed);
#if defined(SMART_WITH_QNN)
    std::string qnn_path = "";
    app.add_option("--qnn-path", qnn_path);
#endif

    CLI11_PARSE(app, argc, argv);

    // get number of CPUs
    {
        auto n_cpus = uv_available_parallelism(); // Default fallback value
        // NOTE: the main thread is also a worker thread, so we need to subtract 1
        SMART_ASSERT(n_cpus >= 2);
        n_threads = std::min((unsigned int)n_threads, n_cpus - 1);
    }

    // get config
    auto config = std::make_shared<smart::Config>(config_path);

    // get platform
    auto platform = std::make_shared<smart::Platform>();
    platform->init_ggml_backend(config, n_threads);
#if defined(SMART_WITH_QNN)
    if (qnn_path != "") {
        platform->init_qnn_backend(qnn_path, config);
    }
#endif

    // get model type
    std::string model_arch = config->arch;
    smart::get_memory_usage("begin");

    std::unique_ptr<smart::Model> model;
    // TODO: move into Model.cpp like build_model
    if (model_arch == "llama") {
        model = std::make_unique<smart::LlamaModel>(file_path, config, platform);
    } else if (model_arch == "phi3") {
        SMART_ASSERT(false);
    } else {
        fmt::print("Unknown model type\n");
    }
    smart::get_memory_usage("after model init");

    if (attn_type == "normal") {
        model->m_attn = std::make_shared<smart::NormAttention>(model->m_config, model->m_weights);
    } else if (attn_type == "quest") {
        model->m_attn = std::make_shared<smart::QuestAttention>(model->m_config, model->m_weights);
        // SMART_ASSERT(false);
    }
    smart::get_memory_usage("after attn init");

    // load tokenizer
    smart::Tokenizer tokenizer(tokenizer_path);
    smart::get_memory_usage("after tokenizer init");

    // load sampler
    smart::SamplerConfig sampler_config{
        .seed            = rng_seed,
        .temp            = temperature,
        .top_p           = top_p,
        .top_k           = top_k,
        .vocab_size      = static_cast<int32_t>(tokenizer.n_vocabs()),
        .special_eos_id  = tokenizer.m_vocab.special_eos_id,
        .linefeed_id     = tokenizer.m_vocab.linefeed_id,
        .penalty_last_n  = 64,
        .penalty_repeat  = 2.0f,
        .penalty_freq    = 1.0f,
        .penalty_present = 0.1f,
        .penalize_nl     = false,
        .ignore_eos      = false,
    };
    smart::SamplerChain sampler{sampler_config};
    smart::get_memory_usage("after sampler init");

    {
        fmt::println(stderr, "file_path   : {}", file_path);
        fmt::println(stderr, "vocab_path  : {}", tokenizer_path);
        fmt::println(stderr, "prompt      : {}", prompt);
        fmt::println(stderr, "steps       : {}", steps);
        fmt::println(stderr, "attn_type   : {}", attn_type);
        fmt::println(stderr, "model arch  : {}", model_arch);
        fmt::println(stderr, "n_threads   : {}", n_threads);
        fmt::println(stderr, "temperature : {}", temperature);
        fmt::println(stderr, "top_p       : {}", top_p);
        fmt::println(stderr, "top_k       : {}", top_k);
        fmt::println(stderr, "rng_seed    : {}", rng_seed);
    }

    // generate
    model->generate(tokenizer, sampler, prompt, steps);
}
