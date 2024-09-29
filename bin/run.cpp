 
#include "model.hpp"
#include "llama_tokenizer.hpp"
#include "sampler.hpp"
#include "generate.hpp"


using namespace smart;

int main() {
    // 0. load config
    std::string file_path = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama2-7b_Q4_0.gguf";
    std::string tokenizer_path = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama2_7b_vocab.gguf";
    float temperature = 1.0f;                // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;                       // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 64;                          // number of steps to run for
    std::string prompt = "One day,";         // prompt string
    unsigned long long rng_seed = 2024927;

    // 1. load model
    Transformer transformer;
    build_transformer(&transformer, file_path);

    // 2. load tokenizer
    smart::LlamaTokenizer tokenizer(tokenizer_path);

    // 3. load sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // 4. generate
    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    // 5. free resources
    free_sampler(&sampler);
    free_transformer(&transformer);
}
