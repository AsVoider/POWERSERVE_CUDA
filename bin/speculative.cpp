
#include "CLI/CLI.hpp"
#include "common.hpp"
#include "model/llama/llama_model.hpp"
#include "model/module/norm_attention.hpp"
#include "model/module/quest_attention.hpp"
#include "sampler/sampler_chain.hpp"
#include "speculative/speculator.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstdlib>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
    // 0. load config
    std::string target_path        = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama-2-7b.f32.gguf";
    std::string draft_path         = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama-2-7b.f32.gguf";
    std::string tokenizer_path     = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama2_7b_vocab.gguf";
    std::string target_config_path = "/home/zwb/SS/smartserving/llama3.2.json";
    std::string draft_config_path  = "/home/zwb/SS/smartserving/llama3.2.json";
    float temperature              = 0.8f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float top_p                    = 0.95f;
    size_t top_k                   = 40;
    int steps                      = 1000; // number of steps to run for
    // std::string prompt         = "One day,"; // prompt string
    std::string prompt =
        "<|start_header_id|>user<|end_header_id|>Mia is planning a camping trip in the Canadian Rockies and has a "
        "budget of $800 for equipment. She buys a tent for $120, which is 15% off the original price. She then "
        "purchases a sleeping bag for $80, which is 20% off. If she also needs to buy a backpack and a portable stove, "
        "and the total cost of these two items is $180, what percentage of her budget will she have left after all the "
        "purchases? <|eot_id|><|start_header_id|>assistant<|end_header_id|>"; // prompt string
    // std::string prompt = "Can you tell me a long story?";
    std::string attn_type = "normal";
    int n_threads         = 4;
    uint64_t rng_seed     = uint64_t(-1); // uint64_t(-1) = random seed

    CLI::App app("Demo program for llama3");

    app.add_option("--target-path", target_path)->required();
    app.add_option("--draft-path", draft_path)->required();
    app.add_option("--vocab-path", tokenizer_path)->required();
    app.add_option("--target-config-path", target_config_path)->required();
    app.add_option("--draft-config-path", draft_config_path)->required();
    app.add_option("--prompt", prompt);
    app.add_option("--steps", steps);
    app.add_option("--attn-type", attn_type);
    app.add_option("--n-threads", n_threads);
    app.add_option("--temperature", temperature);
    app.add_option("--top-p", top_p);
    app.add_option("--top-k", top_k);
    app.add_option("--rng-seed", rng_seed);
#if defined(SMART_WITH_QNN)
    std::string draft_qnn_path  = "";
    std::string target_qnn_path = "";
    app.add_option("--target-qnn-path", target_qnn_path);
    app.add_option("--draft-qnn-path", draft_qnn_path);
#endif

    CLI11_PARSE(app, argc, argv);

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
        // fmt::println(stderr, "file_path   : {}", file_path);
        fmt::println(stderr, "vocab_path  : {}", tokenizer_path);
        fmt::println(stderr, "prompt      : {}", prompt);
        fmt::println(stderr, "steps       : {}", steps);
        fmt::println(stderr, "attn_type   : {}", attn_type);
        // fmt::println(stderr, "model arch  : {}", model_arch);
        fmt::println(stderr, "n_threads   : {}", n_threads);
        fmt::println(stderr, "temperature : {}", temperature);
        fmt::println(stderr, "top_p       : {}", top_p);
        fmt::println(stderr, "top_k       : {}", top_k);
        fmt::println(stderr, "rng_seed    : {}", rng_seed);
    }

    // generate
    // model->generate(tokenizer, sampler, prompt, steps);
    smart::Speculative spec(
        target_path,
        draft_path,
        target_config_path,
        draft_config_path,
        target_qnn_path,
        draft_qnn_path,
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
    );
    spec.generate(tokenizer, sampler, prompt, steps);
}
