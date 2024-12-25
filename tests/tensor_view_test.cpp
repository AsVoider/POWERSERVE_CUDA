#include "CLI/CLI.hpp"
#include "backend/platform.hpp"
#include "common.hpp"
#include "executor/executor.hpp"
#include "graph/node.hpp"
#include "model/model.hpp"
#include "model/model_loader.hpp"

#include <string>
#include <unistd.h>

using namespace smart;

int main(int argc, char *argv[]) {
    std::string config_path = "/home/zwb/SS/smartserving/";

    CLI::App app("Demo program for llama3");

    app.add_option("--config-path", config_path)->required();
    CLI11_PARSE(app, argc, argv);

    auto config                                     = std::make_shared<smart::Config>(config_path);
    std::unique_ptr<smart::Model> model             = smart::load_model(config->main_llm_config, config->main_llm_dir);
    auto [sampler_config, steps, n_threads, prompt] = config->hyper_params;
    model->m_platform                               = std::make_shared<smart::Platform>();
    model->m_platform->init_ggml_backend(model->m_config, n_threads);

    Graph g;

    size_t n_ctx      = 5;
    size_t kv_head    = 4;
    size_t head_size  = 2;
    size_t kv_dim     = kv_head * head_size;
    size_t batch_size = 2;
    size_t cur_pos    = 1;

    auto cache                   = Tensor(smart::DataType::FP32, {n_ctx, kv_dim, 1, 1});
    std::vector<float> cache_buf = std::vector<float>(n_ctx * kv_dim, 0.f);
    // std::iota(cache_buf.begin(), cache_buf.end(), 1.f);
    ggml::Buffer::Stride cache_stride = {
        sizeof(float), sizeof(float) * n_ctx, sizeof(float) * n_ctx * kv_dim, sizeof(float) * n_ctx * kv_dim
    };
    cache.m_data = std::make_shared<ggml::Buffer>(cache_stride, cache_buf.data());

    auto v                   = Tensor(smart::DataType::FP32, {kv_dim, batch_size, 1, 1});
    std::vector<float> v_buf = std::vector<float>(kv_dim * batch_size);
    std::iota(v_buf.begin(), v_buf.end(), 1.f);

    ggml::Buffer::Stride v_stride = {
        sizeof(float), sizeof(float) * kv_dim, sizeof(float) * batch_size * kv_dim, sizeof(float) * batch_size * kv_dim
    };
    v.m_data = std::make_shared<ggml::Buffer>(v_stride, v_buf.data());

    {
        auto Tcache = g.add_tensor(cache);
        g.print(Tcache, -1);
        auto Tv = g.add_tensor(v);
        g.print(Tv, -1);

        auto TVcache = g.view(
            Tcache,
            {batch_size, kv_dim, 1, 1},
            {Tcache->element_size(),
             n_ctx * Tcache->element_size(),
             n_ctx * Tcache->element_size() * kv_dim,
             n_ctx * Tcache->element_size() * kv_dim},
            cur_pos * Tcache->element_size()
        );
        g.print(TVcache, -1);

        auto TPv = g.transpose(Tv);
        g.print(TPv, -1);

        g.copy(TVcache, TPv);
        g.print(TVcache, -1);
        g.print(Tcache, -1);
    }

    Executor executor(*model->m_platform, g);
    executor.allocate_buffers();
    executor.run();
}
