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
#include "core/logger.hpp"
#include "core/timer.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "sampler/sampler_chain.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstdlib>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
    smart::print_timestamp();

    std::string work_folder = "/home/zwb/SS/smartserving/";
    std::string prompt      = "One day,";
    std::string prompt_file = "";
    int n_predicts          = 128;
    bool no_qnn             = false;

    CLI::App app("Demo program for LLM");

    app.add_option("-d,--work-folder", work_folder, "Set the working folder (required).")->required();
    app.add_option("-n,--n_predicts", n_predicts, "Specify the number of predictions to make.");
    app.add_flag("--no-qnn", no_qnn, "Disable QNN processing.");

    CLI::Option_group *prompt_group =
        app.add_option_group("Prompt Options", "Choose either prompt or prompt-file, not both.");
    auto prompt_opt      = prompt_group->add_option("-p,--prompt", prompt);
    auto prompt_file_opt = prompt_group->add_option("-f,--prompt-file", prompt_file);
    prompt_group->require_option(0, 1);

    CLI11_PARSE(app, argc, argv);

    if (prompt_file != "") {
        std::ifstream f(prompt_file);
        if (f.is_open()) {
            std::ostringstream oss;
            oss << f.rdbuf();
            prompt = oss.str();
            f.close();
        } else {
            SMART_ASSERT(false, "failed to open prompt file: {}", prompt_file);
        }
    }

    auto config                         = std::make_shared<smart::Config>(work_folder);
    std::shared_ptr<smart::Model> model = smart::load_model(config->main_model_dir, config->main_model_config);
    SMART_LOG_INFO("after model init: {}", smart::perf_get_mem_result());

    auto [sampler_config, n_threads, batch_size] = config->hyper_params;
    model->m_platform                            = std::make_shared<smart::Platform>();
    model->m_platform->init_ggml_backend(model->m_config, config->hyper_params);
#if defined(SMART_WITH_QNN)
    if (!no_qnn) {
        auto &qnn_backend = model->m_platform->qnn_backend;
        model->m_platform->init_qnn_backend(smart::Path(work_folder) / smart::qnn::QNN_LIB_DIR_NAME);
        qnn_backend->load_model(config->main_model_dir / smart::qnn::QNN_WORKSPACE_DIR_NAME, model->m_config);
    }
#endif
    SMART_LOG_INFO("after platform init: {}", smart::perf_get_mem_result());

    model->m_attn = std::make_shared<smart::NormAttention>(model->m_config->llm, model->m_weights);
    SMART_LOG_INFO("after attn init: {}", smart::perf_get_mem_result());

    std::string tokenizer_path = config->main_model_dir / smart::MODEL_VOCAB_FILENAME;
    smart::Tokenizer tokenizer(tokenizer_path);
    SMART_LOG_INFO("after tokenizer init: {}", smart::perf_get_mem_result());

    smart::SamplerChain sampler{sampler_config, tokenizer};
    SMART_LOG_INFO("after sampler init: {}", smart::perf_get_mem_result());

    {
        SMART_LOG_INFO("prompt      : {:?}", smart::abbreviation(prompt, 50));
        SMART_LOG_INFO("n_predicts       : {}", n_predicts);
        SMART_LOG_INFO("model arch  : {}", config->main_model_config->arch);
        SMART_LOG_INFO("n_threads   : {}", n_threads);
    }

    // generate
    long prefill_start = 0;
    long prefill_end   = 0;
    long decode_end    = 0;
    bool start         = false;
    int actual_predict = 0;
    for (auto prompt_token : tokenizer.tokenize(prompt, tokenizer.m_vocab.tokenizer_add_bos)) {
        fmt::print("{}", tokenizer.to_string(prompt_token, false));
    }
    prefill_start = smart::timestamp_ms();
    for (auto next : model->generate(tokenizer, sampler, prompt, n_predicts, batch_size)) {
        if (!start) {
            prefill_end = smart::timestamp_ms();
            start       = true;
            continue;
        }
        actual_predict += 1;
        if (next == tokenizer.bos_token()) {
            break;
        }
        if (next == tokenizer.m_vocab.special_eos_id || next == tokenizer.m_vocab.special_eom_id ||
            next == tokenizer.m_vocab.special_eot_id) {
            fmt::print("[end of text]");
            break;
        }
        fmt::print("{}", tokenizer.to_string(next, false));
        fflush(stdout);
    }
    fmt::println("");

    if (start) {
        decode_end     = smart::timestamp_ms();
        auto n_prefill = tokenizer.tokenize(prompt, tokenizer.m_vocab.tokenizer_add_bos).size() - 1;
        SMART_LOG_INFO("prefill time: {} s", (double)(prefill_end - prefill_start) / 1000);
        SMART_LOG_INFO(
            "prefill speed ({} tokens): {} tokens/s",
            n_prefill,
            n_prefill / (double)(prefill_end - prefill_start) * 1000
        );
        SMART_LOG_INFO(
            "decode speed ({} tokens): {} tokens/s",
            actual_predict,
            actual_predict / (double)(decode_end - prefill_end) * 1000
        );
        SMART_LOG_INFO(
            "total speed: {} tokens/s", (n_prefill + actual_predict) / (double)(decode_end - prefill_start) * 1000
        );
    }
}
