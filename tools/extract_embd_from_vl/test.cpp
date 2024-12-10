#include "CLI/CLI.hpp"
#include "backend/ggml/ggml.hpp"
#include "common.hpp"
#include "executor/executor.hpp"
#include "ggml.h"
#include "graph/graph.hpp"

int main(int argc, char *argv[]) {
    std::string raw_path = "";
    std::string only_path;
    std::string config_path;

    CLI::App app("Demo program for embedding extracter");
    app.add_option("--only-embd-path", only_path)->required();
    app.add_option("--raw-embd-path", raw_path);
    app.add_option("--config-path", config_path)->required();
    CLI11_PARSE(app, argc, argv);

    ggml_context *ggml_ctx1, *ggml_ctx2 = nullptr;
    gguf_context *gguf_ctx1, *gguf_ctx2 = nullptr;
    {
        gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx1};
        gguf_ctx1               = gguf_init_from_file(only_path.c_str(), params);
        SMART_ASSERT(gguf_ctx1 != nullptr);
        SMART_ASSERT(ggml_ctx1 != nullptr);
    }
    if (raw_path != "") {
        gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx2};
        gguf_ctx2               = gguf_init_from_file(raw_path.c_str(), params);
        SMART_ASSERT(gguf_ctx2 != nullptr);
        SMART_ASSERT(ggml_ctx2 != nullptr);
    }

    auto config   = std::make_shared<smart::Config>(config_path);
    auto platform = std::make_shared<smart::Platform>();
    platform->init_ggml_backend(config, 1);

    smart::Graph g;
    smart::Executor executor(*platform, g);

    std::vector<int> pos{0, 1, 2};

    auto token_embedding_table = smart::ggml::convert_from_ggml(ggml_get_tensor(ggml_ctx1, "token_embd.weight"));
    smart::Tensor raw_embedding_table;
    auto embd_tb1 = g.add_tensor(token_embedding_table);
    auto x1       = g.get_embedding(embd_tb1, pos);

    if (ggml_ctx2) {
        auto raw_embedding_table = smart::ggml::convert_from_ggml(ggml_get_tensor(ggml_ctx2, "token_embd.weight"));

        auto embd_tb2 = g.add_tensor(raw_embedding_table);
        auto x2       = g.get_embedding(embd_tb2, pos);
        g.cos_sim(x1, x2);
    } else {
        g.print(x1, -1);
    }

    executor.allocate_buffers();
    executor.run();

    gguf_free(gguf_ctx1);
    if (gguf_ctx2)
        gguf_free(gguf_ctx2);
}
