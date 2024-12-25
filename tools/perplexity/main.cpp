#include "CLI/CLI.hpp"
#include "backend/platform.hpp"
#include "common/logger.hpp"
#include "common/perf.hpp"
#include "core/config.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"

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
    void accept(smart::Token token);
};

void PerplexityCalculator::apply(smart::ProbArray probs) {
    probs.softmax();
    for (const auto &p : probs.m_probs) {
        log_logits[p.index] = std::log(p.prob);
    }
    n_tokens++;
}

void PerplexityCalculator::accept(smart::Token token) {
    m_logit_sum += log_logits[token];
    current_ppl = std::exp(-m_logit_sum / n_tokens);
}

int main(int argc, char *argv[]) {
    // 0. load config
    std::string work_folder = "/home/zwb/SS/smartserving/llama3.2.json";
    int batch_size          = 32;

    CLI::App app("Demo program for llama3");

    app.add_option("--work-folder", work_folder)->required();
    app.add_option("--batch-size", batch_size);
#if defined(SMART_WITH_QNN)
    bool no_qnn = false;
    app.add_flag("--no-qnn", no_qnn);
#endif

    CLI11_PARSE(app, argc, argv);

    auto config                                     = std::make_shared<smart::Config>(work_folder);
    std::unique_ptr<smart::Model> model             = smart::load_model(config);
    auto [sampler_config, steps, n_threads, prompt] = config->hyper_params;

    model->m_platform = std::make_shared<smart::Platform>();
    model->m_platform->init_ggml_backend(model->m_config, n_threads);
#if defined(SMART_WITH_QNN)
    if (!no_qnn) {
        auto &qnn_backend = model->m_platform->qnn_backend;
        model->m_platform->init_qnn_backend(config->main_model_dir / smart::qnn::QNN_WORKSPACE_DIR_NAME);
        qnn_backend->load_model(config->main_model_dir / smart::qnn::QNN_WORKSPACE_DIR_NAME, model->m_config);
    }
#endif

    model->m_attn = std::make_shared<smart::NormAttention>(model->m_config->llm, model->m_weights);
    SMART_LOG_INFO("after attention module init: {}", smart::perf_get_mem_result());

    // load tokenizer
    std::string tokenizer_path = config->main_model_dir / smart::MODEL_VOCAB_FILENAME;
    smart::Tokenizer tokenizer(tokenizer_path);
    SMART_LOG_INFO("after tokenizer init: {}", smart::perf_get_mem_result());

    // ppl
    PerplexityCalculator ppl_calculator(model->m_config->llm.vocab_size);

    { SMART_LOG_INFO("batch_size  : {}", batch_size); }

    // generate

    auto prompt_tokens = tokenizer.tokenize(config->hyper_params.prompt, tokenizer.m_vocab.tokenizer_add_bos);
    auto n_tokens      = prompt_tokens.size();
    SMART_LOG_INFO("dataset: {} tokens", n_tokens);

    size_t pos      = 0;
    size_t batch_id = 1;
    while (pos < n_tokens) {
        auto size = std::min((size_t)batch_size, n_tokens - pos);
        if (size != batch_size)
            break;
        auto pos_list = std::vector<int>(size);
        std::iota(pos_list.begin(), pos_list.end(), pos);
        std::vector<smart::Token> tokens(size);
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
        if (batch_id >= PPL_START_ID) {
            SMART_LOG_INFO("ppl {}: {}", ppl_calculator.n_tokens, ppl_calculator.current_ppl);
        }
        batch_id += 1;
    }
}
