#include "CLI/CLI.hpp"
#include "backend/platform.hpp"
#include "core/config.hpp"
#include "model/llama/llama_model.hpp"
#include "model/module/norm_attention.hpp"
#include "uv.h"

#include <cmath>
#include <cstddef>
#include <string>

constexpr int PPL_START_ID = 18;

struct PerplexityCalculator {
public:
    size_t n_tokens   = 0;
    float m_logit_sum = 0;
    std::vector<float> log_logits;
    size_t num_vocabs    = 0;
    float_t current_ppl  = 0;
    size_t n_calibration = 1;

public:
    PerplexityCalculator(size_t n_vocabs) : n_tokens(0), m_logit_sum(0), log_logits(), num_vocabs(n_vocabs) {
        log_logits.resize(num_vocabs);
    }

    ~PerplexityCalculator() = default;

public:
    void apply(smart::ProbArray probs);
    void accept(smart::Tokenizer::Token token);
};

void PerplexityCalculator::apply(smart::ProbArray probs) {
    probs.softmax();
    for (const auto &p : probs.m_probs) {
        log_logits[p.index] = std::log(p.prob);
    }
    n_tokens++;
}

void PerplexityCalculator::accept(smart::Tokenizer::Token token) {
    m_logit_sum += log_logits[token];
    current_ppl = std::exp(-m_logit_sum / n_tokens);
}

int main(int argc, char *argv[]) {
    // 0. load config
    std::string file_path      = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama-2-7b.f32.gguf";
    std::string tokenizer_path = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama2_7b_vocab.gguf";
    std::string config_path    = "/home/zwb/SS/smartserving/llama3.2.json";
    std::string prompt_file    = "/home/zwb/SS/libpowerinfer/scripts/converters/datasets/llama3_1.txt";
    int n_threads              = 4;
    int batch_size             = 1;

    CLI::App app("Demo program for llama3");

    app.add_option("--file-path", file_path)->required();
    app.add_option("--vocab-path", tokenizer_path)->required();
    app.add_option("--config-path", config_path)->required();
    app.add_option("--prompt-file", prompt_file)->required();
    app.add_option("--batch-size", batch_size);
    app.add_option("--n-threads", n_threads);
#if defined(SMART_WITH_QNN)
    std::string qnn_path = "";
    app.add_option("--qnn-path", qnn_path);
#endif

    CLI11_PARSE(app, argc, argv);

    // get number of CPUs
    {
        auto n_cpus = uv_available_parallelism(); // Default fallback value
        n_threads   = std::min((unsigned int)n_threads, n_cpus);
    }

    // get config
    auto config = std::make_shared<smart::Config>(config_path);
    fmt::println("config version: {}", config->version);

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
    auto model = std::make_unique<smart::LlamaModel>(file_path, config, platform); // FIXME: use so much time
    smart::get_memory_usage("after model init");

    model->m_attn = std::make_shared<smart::NormAttention>(model->m_config, model->m_weights);
    smart::get_memory_usage("after attn init");

    // load tokenizer
    smart::Tokenizer tokenizer(tokenizer_path);
    smart::get_memory_usage("after tokenizer init");

    // ppl
    PerplexityCalculator ppl_calculator(config->tf_cfg.vocab_size);

    {
        fmt::println(stderr, "file_path   : {}", file_path);
        fmt::println(stderr, "vocab_path  : {}", tokenizer_path);
        fmt::println(stderr, "model arch  : {}", model_arch);
        fmt::println(stderr, "n_threads   : {}", n_threads);
        fmt::println(stderr, "batch_size  : {}", batch_size);
    }

    // generate
    std::string p;
    std::ifstream prompt(prompt_file);
    if (prompt.is_open()) {
        std::string line;
        while (std::getline(prompt, line)) {
            p += line + '\n';
        }
        prompt.close();
    } else {
        fmt::print(stderr, "Error: could not open file {}\n", prompt_file);
        SMART_ASSERT(prompt.is_open());
    }

    auto prompt_tokens = tokenizer.tokenize(p, tokenizer.m_vocab.tokenizer_add_bos);
    auto n_tokens      = prompt_tokens.size();
    fmt::println("dataset: {} tokens", n_tokens);

    size_t pos      = 0;
    size_t batch_id = 1;
    while (pos < n_tokens) {
        auto size = std::min((size_t)batch_size, n_tokens - pos);
        if (size != batch_size)
            break;
        auto pos_list = std::vector<int>(size);
        std::iota(pos_list.begin(), pos_list.end(), pos);
        std::vector<smart::Tokenizer::Token> tokens(size);
        std::copy(prompt_tokens.begin() + pos, prompt_tokens.begin() + pos + size, tokens.begin());
        // decode
        {
            auto mask = smart::CausalAttentionMask(tokens.size());
            auto ret  = model->forward(tokens, pos_list, mask, true);
            for (auto logits : ret) {
                auto probs = smart::ProbArray(logits);
                if (batch_id >= PPL_START_ID) {
                    if (pos != (PPL_START_ID - 1) * batch_size) { // skip the first token
                        ppl_calculator.apply(probs);
                        ppl_calculator.accept(prompt_tokens[pos + 1]);
                    }
                }
                pos += 1;
            }
        }
        if (batch_id >= PPL_START_ID)
            fmt::println(stderr, "ppl {}: {}", ppl_calculator.n_tokens, ppl_calculator.current_ppl);
        batch_id += 1;
    }
}
