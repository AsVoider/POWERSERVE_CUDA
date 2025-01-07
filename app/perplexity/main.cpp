// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "CLI/CLI.hpp"
#include "backend/platform.hpp"
#include "core/config.hpp"
#include "core/logger.hpp"
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
    void apply(powerserve::ProbArray probs);
    void accept(powerserve::Token token);
};

void PerplexityCalculator::apply(powerserve::ProbArray probs) {
    probs.softmax();
    for (const auto &p : probs.m_probs) {
        log_logits[p.token] = std::log(p.prob);
    }
    n_tokens++;
}

void PerplexityCalculator::accept(powerserve::Token token) {
    m_logit_sum += log_logits[token];
    current_ppl = std::exp(-m_logit_sum / n_tokens);
}

int main(int argc, char *argv[]) {
    // 0. load config
    std::string work_folder;
    std::string prompt_file;
    std::string prompt;
    size_t batch_size = 32;
    bool no_qnn       = false;

    CLI::App app("Perplexity");

    app.add_option("-d,--work-folder", work_folder, "Set the working folder (required).")->required();
    app.add_option("-b,--batch-size", batch_size);
    app.add_flag("--no-qnn", no_qnn, "Disable QNN processing.");

    CLI::Option_group *prompt_group =
        app.add_option_group("Prompt Options", "Choose either prompt or prompt-file, not both.");
    prompt_group->add_option("-p,--prompt", prompt);
    prompt_group->add_option("-f,--prompt-file", prompt_file);
    prompt_group->require_option(1, 1);

    CLI11_PARSE(app, argc, argv);

    if (!prompt_file.empty()) {
        std::ifstream f(prompt_file);
        if (f.is_open()) {
            std::ostringstream oss;
            oss << f.rdbuf();
            prompt = oss.str();
            f.close();
        } else {
            POWERSERVE_ASSERT(false, "failed to open prompt file: {}", prompt_file);
        }
    }
    auto config = std::make_shared<powerserve::Config>(work_folder);
    std::shared_ptr<powerserve::Model> model =
        powerserve::load_model(config->main_model_dir, config->main_model_config);
    auto [sampler_config, n_threads, prefill_batch_size] = config->hyper_params;
    POWERSERVE_UNUSED(prefill_batch_size);

    model->m_platform = std::make_shared<powerserve::Platform>();
    model->m_platform->init_ggml_backend(model->m_config, config->hyper_params);
#if defined(POWERSERVE_WITH_QNN)
    if (!no_qnn) {
        auto &qnn_backend = model->m_platform->qnn_backend;
        model->m_platform->init_qnn_backend(powerserve::Path(work_folder) / powerserve::qnn::QNN_LIB_DIR_NAME);
        qnn_backend->load_model(config->main_model_dir / powerserve::qnn::QNN_WORKSPACE_DIR_NAME, model->m_config);
    }
#endif

    model->m_attn = std::make_shared<powerserve::NormAttention>(model->m_config->llm, model->m_weights);
    POWERSERVE_LOG_INFO("after attention module init: {}", powerserve::perf_get_mem_result());

    // load tokenizer
    const std::string tokenizer_path = config->main_model_dir / powerserve::MODEL_VOCAB_FILENAME;
    const powerserve::Tokenizer tokenizer(tokenizer_path);
    POWERSERVE_LOG_INFO("after tokenizer init: {}", powerserve::perf_get_mem_result());

    const auto prompt_tokens = tokenizer.tokenize(prompt, tokenizer.m_vocab.tokenizer_add_bos);
    const size_t n_tokens    = prompt_tokens.size();
    POWERSERVE_LOG_INFO("dataset: {} tokens", n_tokens);

    // ppl
    PerplexityCalculator ppl_calculator(model->m_config->llm.vocab_size);

    if (batch_size > n_tokens) {
        batch_size = n_tokens;
    }
    { POWERSERVE_LOG_INFO("batch_size  : {}", batch_size); }

    POWERSERVE_ASSERT(n_tokens >= batch_size * PPL_START_ID);

    // generate
    size_t pos      = 0;
    size_t batch_id = 1;
    while (pos < n_tokens) {
        const size_t size = std::min(batch_size, n_tokens - pos);
        if (size != batch_size) {
            break;
        }

        std::vector<int> pos_list(size);
        std::iota(pos_list.begin(), pos_list.end(), pos);
        std::vector<powerserve::Token> tokens(size);
        std::copy(prompt_tokens.begin() + pos, prompt_tokens.begin() + pos + size, tokens.begin());
        // decode
        {
            const powerserve::CausalAttentionMask mask(tokens.size());
            const auto ret = model->forward(tokens, pos_list, mask, true);
            for (auto logits : ret.logits_vector) {
                auto probs = powerserve::ProbArray(logits);
                if (batch_id >= PPL_START_ID && pos + 1 < n_tokens) {
                    if (pos != (PPL_START_ID - 1) * batch_size) { // skip the first token
                        ppl_calculator.apply(probs);
                        ppl_calculator.accept(prompt_tokens[pos + 1]);
                    }
                }
                pos += 1;
            }
        }
        if (batch_id >= PPL_START_ID) {
            POWERSERVE_LOG_INFO("ppl {}: {}", ppl_calculator.n_tokens, ppl_calculator.current_ppl);
        }
        batch_id += 1;
    }
}
