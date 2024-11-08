#include "backend/platform.hpp"
#include "executor/executor.hpp"
#include "common.hpp"
#include "ggml.h"
#include "graph/node.hpp"
#include "model/llama/llama_config.hpp"
#include <string>
#include <unistd.h>

using namespace smart;

int main() {
    std::string filename   = "../models/Meta-Llama-3.1-8B/llama3-8b_Q4_0.gguf";
    ggml_context *ggml_ctx = nullptr;
    gguf_context *gguf_ctx = nullptr;

    TensorNode *a, *b, *out;
    int n_round  = 2;
    size_t dim1 = 2;
    size_t dim2 = 2;
    Graph g;
    std::shared_ptr<Platform> plat     = nullptr;
    std::shared_ptr<Executor> executor = nullptr;

    {
        gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx};
        gguf_ctx                = gguf_init_from_file(filename.c_str(), params);
        SMART_ASSERT(gguf_ctx != nullptr);
        SMART_ASSERT(ggml_ctx != nullptr);
    }

    std::shared_ptr<LlamaConfig> m_config = std::make_shared<LlamaConfig>(gguf_ctx);

    a   = g.new_tensor(DataType::FP32, {dim1 * dim2});
    b   = g.new_tensor(DataType::FP32, {dim1 * dim2});
    auto a_view = g.view_tensor(a, {dim1, dim2, 1, 1});
    auto b_view = g.view_tensor(b, {dim1, dim2, 1, 1});

    out = g.add(a_view, b_view);
    // for (int n = 1; n < n_round; n++)
    // out = g.mat_mul(a_view, out);

    plat     = std::make_shared<Platform>(m_config, 4);
    executor = std::make_shared<Executor>(*plat, g);
    executor->allocate_buffers();

    {
        auto a_buf = static_cast<float *>(a->get<ggml::Buffer>().m_data);
        auto b_buf = static_cast<float *>(b->get<ggml::Buffer>().m_data);
        for (int j = 0; j < a->m_shape[1]; j++)
            for (int i = 0; i < a->m_shape[0]; i++) {
                a_buf[j * a->m_shape[0] + i] = i * 1.0 - j;
                b_buf[j * b->m_shape[0] + i] = i * 2.0 + j;
                fmt::println("a[{}][{}] {}", j, i, a_buf[j * a->m_shape[0] + i]);
                fmt::println("b[{}][{}] {}", j, i, b_buf[j * b->m_shape[0] + i]);
            }
    }

    executor->run();

    {
        auto out_buf = static_cast<float *>(out->get<ggml::Buffer>().m_data);
        for (int j = 0; j < out->m_shape[1]; j++)
            for (int i = 0; i < out->m_shape[0] ; i++) {
                fmt::println("c[{}][{}] {}", j, i, out_buf[j * out->m_shape[0] + i]);
            }
    }

    { gguf_free(gguf_ctx); }
}