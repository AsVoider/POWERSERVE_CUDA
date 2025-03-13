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

#include "platform.hpp"

namespace powerserve {

std::unordered_map<TensorBackend, BufferInterface> Platform::buffer_interfaces{};

void Platform::init_backend(
    const std::shared_ptr<ModelConfig> &config, const HyperParams &hparams, [[maybe_unused]] const Path &qnn_path
) {
    backends[config->model_id].insert(
        std::make_pair(TensorBackend::GGML_CPU, std::make_unique<ggml::GGMLBackend>(config->llm, hparams))
    );
    buffer_interfaces.insert(std::make_pair(
        TensorBackend::GGML_CPU,
        BufferInterface{
            .create_buffer      = ggml::CPUBuffer::create_buffer,
            .create_buffer_view = ggml::CPUBuffer::create_buffer_view,
        }
    ));

#if defined(POWERSERVE_WITH_CUDA)
    backends[config->model_id].insert(
        std::make_pair(TensorBackend::GGML_GPU, std::make_unique<ggml_cuda::GGML_CUDABackend>(config->llm, hparams))
    );
    buffer_interfaces.insert(std::make_pair(
        TensorBackend::GGML_GPU,
        BufferInterface{
            .create_buffer      = ggml_cuda::Buffer_CUDA::create_buffer,
            .create_buffer_view = ggml_cuda::Buffer_CUDA::create_buffer_view,
        }
    ));
#endif

#if defined(POWERSERVE_WITH_QNN)
    if (qnn_backend) {
        qnn_backend = std::make_unique<qnn::QNNBackend>(qnn_path);
    }
#endif
}

void Platform::destroy_backend(const std::shared_ptr<ModelConfig> &config) {
    backends[config->model_id].clear();
    backends.erase(config->model_id);
}

#if defined(POWERSERVE_WITH_QNN)
void Platform::init_qnn_backend(const Path &qnn_path) {
    qnn_backend = std::make_unique<qnn::QNNBackend>(qnn_path);
}
#endif

size_t Platform::get_kv_position(std::string &model_id) const {
    // NEW ADD
    auto position{
        static_cast<ggml::GGMLBackend &>(*backends.at(model_id).at(TensorBackend::GGML_CPU)).m_kv->get_cache_position()
    };

#if defined(POWERSERVE_WITH_CUDA)
    auto cuda_position{static_cast<ggml_cuda::GGML_CUDABackend &>(*backends.at(model_id).at(TensorBackend::GGML_GPU))
                           .m_kv->get_cache_position()};
    POWERSERVE_ASSERT(cuda_position == position);
#endif

#if defined(POWERSERVE_WITH_QNN)
    if (qnn_backend) {
        auto qnn_position{qnn_backend->m_models[model_id]->kv_cache->position};
        POWERSERVE_ASSERT(qnn_position == position);
        `
    }
#endif
    return position;
}

void Platform::reset_kv_position(std::string &model_id) {
    // ggml_backends[model_id]->m_kv->reset_kv_cache();
    static_cast<ggml::GGMLBackend &>(*backends.at(model_id).at(TensorBackend::GGML_CPU)).m_kv->clear_cache(0UL);
#if defined(POWERSERVE_WITH_QNN)
    if (qnn_backend) {
        qnn_backend->m_models[model_id]->reset_kv_cache();
    }
#endif
}

} // namespace powerserve
