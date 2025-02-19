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
#include "core/perf.hpp"
#include "core/perfetto_trace.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "sampler/sampler_chain.hpp"
#include "speculative/tree_speculative.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstdlib>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
    // 0. load config
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
    prompt_group->add_option("-p,--prompt", prompt);
    prompt_group->add_option("-f,--prompt-file", prompt_file);
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

    auto config                               = std::make_shared<smart::Config>(work_folder);
    std::shared_ptr<smart::Model> main_model  = smart::load_model(config->main_model_dir, config->main_model_config);
    std::shared_ptr<smart::Model> draft_model = smart::load_model(config->draft_model_dir, config->draft_model_config);
    auto [sampler_config, n_threads, batch_size] = config->hyper_params;

    main_model->m_platform  = std::make_shared<smart::Platform>();
    auto &platform          = main_model->m_platform;
    draft_model->m_platform = platform;

    platform->init_ggml_backend(main_model->m_config, config->hyper_params);
    platform->init_ggml_backend(draft_model->m_config, config->hyper_params);
#if defined(SMART_WITH_QNN)
    if (!no_qnn) {
        platform->init_qnn_backend(smart::Path(work_folder) / smart::qnn::QNN_LIB_DIR_NAME);
        auto &qnn_backend = platform->qnn_backend;
        qnn_backend->load_model(config->main_model_dir / smart::qnn::QNN_WORKSPACE_DIR_NAME, main_model->m_config);
        qnn_backend->load_model(config->draft_model_dir / smart::qnn::QNN_WORKSPACE_DIR_NAME, draft_model->m_config);

        // TODO: Fix it.
        main_model->kv_cache  = platform->qnn_backend->m_models[main_model->m_config->model_id]->kv_cache.get();
        draft_model->kv_cache = platform->qnn_backend->m_models[draft_model->m_config->model_id]->kv_cache.get();
    }
#endif

    main_model->m_attn  = std::make_shared<smart::NormAttention>(main_model->m_config->llm, main_model->m_weights);
    draft_model->m_attn = std::make_shared<smart::NormAttention>(draft_model->m_config->llm, draft_model->m_weights);

    std::string tokenizer_path = config->main_model_dir / smart::MODEL_VOCAB_FILENAME;
    smart::Tokenizer tokenizer(tokenizer_path);
    SMART_LOG_INFO("after tokenizer init: {}", smart::perf_get_mem_result());

    smart::SamplerChain sampler{sampler_config, tokenizer};
    SMART_LOG_INFO("after sampler init: {}", smart::perf_get_mem_result());

    {
        SMART_LOG_INFO("prompt      : {}", smart::abbreviation(prompt, 50));
        SMART_LOG_INFO("vocab_path  : {}", tokenizer_path);
        SMART_LOG_INFO("n_predicts  : {}", n_predicts);
        SMART_LOG_INFO("n_threads   : {}", n_threads);
    }

    smart::PerfettoTrace::instance().start_tracing(32 * 1024);
    smart::TreeSpeculative spec(main_model, draft_model);
    spec.generate(tokenizer, sampler, prompt, n_predicts);
    spec.print_stat();
    smart::PerfettoTrace::instance().stop_tracing();
}
