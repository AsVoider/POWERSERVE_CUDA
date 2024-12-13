#include "CLI/CLI.hpp"
#include "backend/platform.hpp"
#include "common.hpp"
#include "core/config.hpp"
#include "crow.h"
#include "model/model.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "model/module/quest_attention.hpp"
#include "sampler/sampler_chain.hpp"

#include <cstdlib>

int main(int argc, char *argv[]) {
    smart::print_timestamp();

    // 0. load config
    std::string config_path = "/home/zwb/SS/smartserving/llama3.2.json";
    std::string attn_type   = "normal";
    std::string host        = "127.0.0.1";
    int port                = 8080;

    CLI::App app("Server program");

    app.add_option("--config-path", config_path)->required();
    app.add_option("--attn-type", attn_type);
    app.add_option("--host", host);
    app.add_option("--port", port);
#if defined(SMART_WITH_QNN)
    bool use_qnn = false;
    app.add_flag("--use-qnn", use_qnn);
#endif

    CLI11_PARSE(app, argc, argv);

    auto config                         = std::make_shared<smart::Config>(config_path);
    std::unique_ptr<smart::Model> model = smart::load_model(config->main_llm_config, config->main_llm_dir);

    model->m_platform = std::make_shared<smart::Platform>();
    model->m_platform->init_ggml_backend(model->m_config, config->hyper_params.n_threads);
#if defined(SMART_WITH_QNN)
    if (use_qnn) {
        model->m_platform->init_qnn_backend(
            config->main_llm_dir / smart::qnn::QNN_WORKSPACE_DIR_NAME, config->main_llm_config
        );
    }
#endif

    if (attn_type == "normal") {
        model->m_attn = std::make_shared<smart::NormAttention>(model->m_config, model->m_weights);
    } else if (attn_type == "quest") {
        model->m_attn = std::make_shared<smart::QuestAttention>(model->m_config, model->m_weights);
    }
    smart::get_memory_usage("after attn init");

    std::string tokenizer_path = config->main_llm_dir / smart::LLM_VOCAB_FILENAME;
    smart::Tokenizer tokenizer(tokenizer_path);
    smart::get_memory_usage("after tokenizer init");

    crow::SimpleApp server;
    server.bindaddr(host).port(port);

    CROW_ROUTE(server, "/completion").methods("POST"_method)([&](const crow::request &req) {
        auto body = crow::json::load(req.body);
        if (!body)
            return crow::response(400);

        auto prompt         = std::string(body["prompt"].s());
        auto n_predict      = body["n_predict"].i();
        auto temperature    = body["temperature"].d();
        auto repeat_penalty = body["repeat_penalty"].d();

        auto &sampler_config          = config->hyper_params.sampler_config;
        sampler_config.temperature    = temperature;
        sampler_config.penalty_repeat = repeat_penalty;
        smart::SamplerChain sampler{sampler_config, tokenizer};

        crow::json::wvalue resp;

        std::stringstream ss;
        auto start = 0;
        for (auto next : model->generate(tokenizer, sampler, prompt, n_predict)) {
            if (start == 0) {
                start = 1; // filter the last prompt token
                continue;
            }
            if (next == tokenizer.bos_token()) {
                continue;
            }
            if (next == tokenizer.m_vocab.special_eos_id || next == tokenizer.m_vocab.special_eom_id ||
                next == tokenizer.m_vocab.special_eot_id) {
                ss << "[end of text]";
                break;
            }
            ss << tokenizer.to_string(next);
        }
        resp["content"] = ss.str();

        smart::get_memory_usage("after chat");
        return crow::response(resp);
    });

    fmt::println("server is running at http://{}:{}", host, port);
    server.run();
}
