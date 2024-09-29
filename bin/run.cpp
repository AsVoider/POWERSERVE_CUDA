 
#include "model.hpp"
#include "llama_tokenizer.hpp"
#include "sampler.hpp"
#include "generate.hpp"
#include "debug.hpp"
#include "CLI/CLI.hpp"


using namespace smart;

int main(int argc, char *argv[]) {

    // 0. load config
    std::string file_path = "../model/Meta-Llama-3.1-8B/llama3-8b_Q4_0.gguf";
    std::string tokenizer_path = "../model/Meta-Llama-3.1-8B/llama3.1_8b_vocab.gguf";
    float temperature = 1.0f;                // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;                       // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 64;                          // number of steps to run for
    std::string prompt = "One day,";         // prompt string
    unsigned long long rng_seed = 2024927;

    CLI::App app("Demo program for llama3");
    
    app.add_option("--file-path", file_path)->required();
    app.add_option("--vocab-path", tokenizer_path)->required();
    app.add_option("--prompt", prompt)->required();
    app.add_option("--steps", steps)->required();
    CLI11_PARSE(app, argc, argv);

    // 1. load model
    Transformer transformer;
    build_transformer(&transformer, file_path);

    // 2. load tokenizer
    smart::LlamaTokenizer tokenizer(tokenizer_path);

    // 3. load sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    {
        debug_meta_info(transformer.gguf_ctx, transformer.ggml_ctx);
        debug_tensors_info(transformer.gguf_ctx, transformer.ggml_ctx);
        debug_config_info(&transformer.config);
        debug_weights_info(&transformer.weights);
    }

    // 4. generate
    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    // 5. free resources
    free_sampler(&sampler);
    free_transformer(&transformer);
}
