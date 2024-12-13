#include "CLI/CLI.hpp"
#include "common.hpp"
#include "model/model_loader.hpp"
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
    std::string attn_type   = "normal";
    std::string config_path = "/home/zwb/SS/smartserving/";

    CLI::App app("Demo program for speculative");

    app.add_option("--config-path", config_path)->required();
    app.add_option("--attn-type", attn_type);
#if defined(SMART_WITH_QNN)
    bool use_qnn = true;
    app.add_flag("--use-qnn", use_qnn);
#endif

    CLI11_PARSE(app, argc, argv);

    auto config                               = std::make_shared<smart::Config>(config_path);
    std::unique_ptr<smart::Model> main_model  = smart::load_model(config->main_llm_config, config->main_llm_dir);
    std::unique_ptr<smart::Model> draft_model = smart::load_model(config->draft_llm_config, config->draft_llm_dir);
    auto [sampler_config, steps, n_threads, prompt] = config->hyper_params;

    main_model->m_platform  = std::make_shared<smart::Platform>();
    draft_model->m_platform = std::make_shared<smart::Platform>();

    main_model->m_platform->init_ggml_backend(main_model->m_config, n_threads);
    draft_model->m_platform->init_ggml_backend(draft_model->m_config, n_threads);
#if defined(SMART_WITH_QNN)
    if (use_qnn) {
        main_model->m_platform->init_qnn_backend(
            config->main_llm_dir / smart::qnn::QNN_WORKSPACE_DIR_NAME, main_model->m_config
        );
        draft_model->m_platform->init_qnn_backend(
            config->draft_llm_dir / smart::qnn::QNN_WORKSPACE_DIR_NAME, draft_model->m_config
        );
    }
#endif

    if (attn_type == "normal") {
        main_model->m_attn  = std::make_shared<smart::NormAttention>(main_model->m_config, main_model->m_weights);
        draft_model->m_attn = std::make_shared<smart::NormAttention>(draft_model->m_config, draft_model->m_weights);
    } else if (attn_type == "quest") {
        main_model->m_attn  = std::make_shared<smart::QuestAttention>(main_model->m_config, main_model->m_weights);
        draft_model->m_attn = std::make_shared<smart::QuestAttention>(draft_model->m_config, draft_model->m_weights);
    }

    std::string tokenizer_path = config->main_llm_dir / smart::LLM_VOCAB_FILENAME;
    smart::Tokenizer tokenizer(tokenizer_path);
    smart::get_memory_usage("after tokenizer init");

    smart::SamplerChain sampler{sampler_config, tokenizer};
    smart::get_memory_usage("after sampler init");

    {
        fmt::println("prompt      : {}", smart::abbreviation(prompt, 50));
        fmt::println(stderr, "vocab_path  : {}", tokenizer_path);
        fmt::println(stderr, "steps       : {}", steps);
        fmt::println(stderr, "attn_type   : {}", attn_type);
        fmt::println(stderr, "n_threads   : {}", n_threads);
    }

    // generate
#if defined(SMART_WITH_QNN)
    smart::Speculative spec(
        std::move(main_model), std::move(draft_model), {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
    );
    spec.generate(tokenizer, sampler, prompt, steps);
#endif
}
