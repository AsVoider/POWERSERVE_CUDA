#include "CLI/CLI.hpp"
#include "backend/platform.hpp"
#include "common.hpp"
#include "core/config.hpp"
#include "crow.h"
#include "model/llama/llama_model.hpp"
#include "model/model.hpp"
#include "model/module/norm_attention.hpp"
#include "model/module/quest_attention.hpp"
#include "sampler/sampler_chain.hpp"

#include <cstdlib>

int main(int argc, char *argv[]) {
    // 0. load config
    std::string file_path      = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama-2-7b.f32.gguf";
    std::string tokenizer_path = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama2_7b_vocab.gguf";
    std::string config_path    = "/home/zwb/SS/smartserving/llama3.2.json";
    float temperature          = 0.8f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float top_p                = 0.95f;
    size_t top_k               = 40;
    std::string attn_type      = "normal";
    int n_threads              = 4;
    uint64_t rng_seed          = uint64_t(-1); // uint64_t(-1) = random seed
    std::string host           = "127.0.0.1";
    int port                   = 8080;

    CLI::App app("Server program");

    app.add_option("--file-path", file_path)->required();
    app.add_option("--vocab-path", tokenizer_path)->required();
    app.add_option("--config-path", config_path)->required();
    app.add_option("--attn-type", attn_type);
    app.add_option("--n-threads", n_threads);
    app.add_option("--temperature", temperature);
    app.add_option("--top-p", top_p);
    app.add_option("--top-k", top_k);
    app.add_option("--rng-seed", rng_seed);
    app.add_option("--host", host);
    app.add_option("--port", port);
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

    std::shared_ptr<smart::Model> model;
    // TODO: move into Model.cpp like build_model
    if (model_arch == "llama") {
        model = std::make_shared<smart::LlamaModel>(file_path, config, platform);
    } else if (model_arch == "phi3") {
        SMART_ASSERT(false);
    } else {
        fmt::print("Unknown model type\n");
    }
    smart::get_memory_usage("after model init");

    if (attn_type == "normal") {
        model->m_attn = std::make_shared<smart::NormAttention>(model->m_config, model->m_weights);
    } else if (attn_type == "quest") {
        model->m_attn = std::make_shared<smart::QuestAttention>(model->m_config, model->m_weights);
        // SMART_ASSERT(false);
    }
    smart::get_memory_usage("after attn init");

    // load tokenizer
    smart::Tokenizer tokenizer(tokenizer_path);
    smart::get_memory_usage("after tokenizer init");

    // load sampler
    smart::SamplerConfig sampler_config{
        .seed            = rng_seed,
        .temp            = temperature,
        .top_p           = top_p,
        .top_k           = top_k,
        .vocab_size      = static_cast<int32_t>(tokenizer.n_vocabs()),
        .special_eos_id  = tokenizer.m_vocab.special_eos_id,
        .linefeed_id     = tokenizer.m_vocab.linefeed_id,
        .penalty_last_n  = 64,
        .penalty_repeat  = 2.0f,
        .penalty_freq    = 1.0f,
        .penalty_present = 0.1f,
        .penalize_nl     = false,
        .ignore_eos      = false,
    };

    crow::SimpleApp server;
    server.bindaddr(host).port(port);

    CROW_ROUTE(server, "/completion").methods("POST"_method)([&](const crow::request &req) {
        // fmt::println(stderr, "{}", req.headers);
        // fmt::println(stderr, "{}", req.body);
        auto body = crow::json::load(req.body);
        if (!body)
            return crow::response(400);

        auto prompt         = std::string(body["prompt"].s());
        auto n_predict      = body["n_predict"].i();
        auto temperature    = body["temperature"].d();
        auto repeat_penalty = body["repeat_penalty"].d();
        // fmt::println(stderr, "prompt        : {}", prompt);
        // fmt::println(stderr, "n_predict     : {}", n_predict);
        // fmt::println(stderr, "temperature   : {}", temperature);
        // fmt::println(stderr, "repeat_penalty: {}", repeat_penalty);

        sampler_config.temp           = temperature;
        sampler_config.penalty_repeat = repeat_penalty;
        smart::SamplerChain sampler{sampler_config};

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

    server.run();
}
