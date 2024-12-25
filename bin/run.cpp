#include "CLI/CLI.hpp"
#include "common/logger.hpp"
#include "common/perf.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "model/module/quest_attention.hpp"
#include "sampler/sampler_chain.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstdlib>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
    smart::print_timestamp();

    std::string work_folder = "/home/zwb/SS/smartserving/";
    std::string attn_type   = "normal";
    std::string prompt      = "";

    CLI::App app("Demo program for llama3");

    app.add_option("--work-folder", work_folder)->required();
    app.add_option("--attn-type", attn_type);
    app.add_option("--prompt", prompt);
#if defined(SMART_WITH_QNN)
    bool no_qnn = false;
    app.add_flag("--no-qnn", no_qnn);
#endif

    CLI11_PARSE(app, argc, argv);

    auto config                                          = std::make_shared<smart::Config>(work_folder);
    std::unique_ptr<smart::Model> model                  = smart::load_model(config);
    auto [sampler_config, steps, n_threads, file_prompt] = config->hyper_params;
    if (prompt == "") {
        prompt = file_prompt;
    }
    model->m_platform = std::make_shared<smart::Platform>();
    model->m_platform->init_ggml_backend(model->m_config, n_threads);
#if defined(SMART_WITH_QNN)
    if (!no_qnn) {
        auto &qnn_backend = model->m_platform->qnn_backend;
        model->m_platform->init_qnn_backend(config->main_model_dir / smart::qnn::QNN_WORKSPACE_DIR_NAME);
        qnn_backend->load_model(config->main_model_dir / smart::qnn::QNN_WORKSPACE_DIR_NAME, model->m_config);
    }
#endif

    if (attn_type == "normal") {
        model->m_attn = std::make_shared<smart::NormAttention>(model->m_config->llm, model->m_weights);
    } else if (attn_type == "quest") {
        model->m_attn = std::make_shared<smart::QuestAttention>(model->m_config->llm, model->m_weights);
    }
    SMART_LOG_INFO("after attn init: {}", smart::perf_get_mem_result());

    std::string tokenizer_path = config->main_model_dir / smart::MODEL_VOCAB_FILENAME;
    smart::Tokenizer tokenizer(tokenizer_path);
    SMART_LOG_INFO("after tokenizer init: {}", smart::perf_get_mem_result());

    smart::SamplerChain sampler{sampler_config, tokenizer};
    SMART_LOG_INFO("after sampler init: {}", smart::perf_get_mem_result());

    {
        SMART_LOG_INFO("prompt      : {}", smart::abbreviation(prompt, 50));
        SMART_LOG_INFO("steps       : {}", steps);
        SMART_LOG_INFO("attn_type   : {}", attn_type);
        SMART_LOG_INFO("model arch  : {}", config->main_model_config->arch);
        SMART_LOG_INFO("n_threads   : {}", n_threads);
    }

    // generate
    long prefill_start = 0;
    long prefill_end   = 0;
    long decode_end    = 0;
    bool start         = false;
    int actual_predict = 0;

    prefill_start = smart::time_in_ms();
    for (auto next : model->generate(tokenizer, sampler, prompt, steps)) {
        if (!start) {
            prefill_end = smart::time_in_ms();
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
        fmt::print("{}", tokenizer.to_string(next));
        fflush(stdout);
    }
    fmt::println("");
    if (start) {
        decode_end     = smart::time_in_ms();
        auto n_prefill = tokenizer.tokenize(prompt, tokenizer.m_vocab.tokenizer_add_bos).size() - 1;
        SMART_LOG_INFO("prefill time: {} s", (double)(prefill_end - prefill_start) / 1000);
        SMART_LOG_INFO("prefill speed: {} tokens/s", n_prefill / (double)(prefill_end - prefill_start) * 1000);
        SMART_LOG_INFO("decode speed: {} tokens/s", actual_predict / (double)(decode_end - prefill_end) * 1000);
        SMART_LOG_INFO(
            "total speed: {} tokens/s", (n_prefill + actual_predict) / (double)(decode_end - prefill_start) * 1000
        );
    }
}
