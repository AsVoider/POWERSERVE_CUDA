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

#include "backend/platform.hpp"
#include "common.hpp"
#include "executor/executor.hpp"
#include "ggml.h"
#include "graph/node.hpp"
#include "model/llama/llama_config.hpp"

#include <string>
#include <unistd.h>

using namespace powerserve;

int main() {

    std::string filename   = "../models/Meta-Llama-3.1-8B/llama3-8b_Q4_0.gguf";
    ggml_context *ggml_ctx = nullptr;
    gguf_context *gguf_ctx = nullptr;

    TensorNode *a, *b, *out;
    int n_round = 1;
    size_t dim1 = 4;
    size_t dim2 = 2;
    Graph g;
    std::shared_ptr<Platform> plat     = nullptr;
    std::shared_ptr<Executor> executor = nullptr;

    {
        gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx};
        gguf_ctx                = gguf_init_from_file(filename.c_str(), params);
        POWERSERVE_ASSERT(gguf_ctx != nullptr);
        POWERSERVE_ASSERT(ggml_ctx != nullptr);
    }

    std::shared_ptr<LlamaConfig> m_config = std::make_shared<LlamaConfig>(gguf_ctx);

    a = g.new_tensor(DataType::FP32, {dim1, dim2});
    b = g.new_tensor(DataType::FP32, {dim1, dim2});
    // out = g.add(a, b);
    out = g.mat_mul(a, b);

    for (int n = 1; n < n_round; n++)
        out = g.add(a, out);

    plat     = std::make_shared<Platform>(m_config, 4);
    executor = std::make_shared<Executor>(*plat, g);
    executor->allocate_buffers();

    sleep(2);
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
            for (int i = 0; i < out->m_shape[0]; i++) {
                fmt::println("c[{}][{}] {}", j, i, out_buf[j * out->m_shape[0] + i]);
            }
    }

    { gguf_free(gguf_ctx); }
}
