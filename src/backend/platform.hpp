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

// 1 platform contains N backends (CPU, NPU, GPU...)
#pragma once

#include "backend/ggml/ggml.hpp"

#include <functional>
#include <map>
#include <unordered_map>

#if defined(POWERSERVE_WITH_QNN)
#include "backend/qnn/qnn_backend.hpp"
#endif

#ifdef POWERSERVE_WITH_CUDA
#include "backend/ggml-cuda/ggml-cuda.hpp"
#endif

namespace powerserve {

using buffer_create_fn   = std::function<BufferPtr(Shape, DataType)>;
using buffer_view_fn     = std::function<BufferPtr(BaseBuffer &, Shape, DataType, size_t)>;
using buffer_alloc_total = std::function<void(size_t)>;

class BufferInterface {
public:
    buffer_create_fn create_buffer;
    buffer_view_fn create_buffer_view;
    buffer_alloc_total alloc_total;
};

struct Platform {
#if defined(POWERSERVE_WITH_QNN)
    std::unique_ptr<qnn::QNNBackend> qnn_backend = nullptr;
#endif
    std::unordered_map<std::string, std::unordered_map<TensorBackend, std::unique_ptr<Backend>>> backends{};
    static std::unordered_map<TensorBackend, BufferInterface> buffer_interfaces;

public:
    Platform() = default;

    ~Platform() = default;

public:
    // TODO: No need trans config
    void init_backend(
        const std::shared_ptr<ModelConfig> &config, const HyperParams &hparams, [[maybe_unused]] const Path &qnn_path
    );
    void destroy_backend(const std::shared_ptr<ModelConfig> &config);
#if defined(POWERSERVE_WITH_QNN)
    void init_qnn_backend(const Path &qnn_path);
#endif

    size_t get_kv_position(std::string &model_id) const;
    void reset_kv_position(std::string &model_id);
};

} // namespace powerserve
