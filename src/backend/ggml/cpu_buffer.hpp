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

#pragma once

#include "core/buffer.hpp"
#include "core/logger.hpp"
#include "core/typedefs.hpp"

namespace powerserve::ggml {

struct CPUBuffer : BaseBuffer {
public:
    CPUBuffer(Stride stride, void *data, bool allocated_by_malloc, size_t size, usage use)
        : BaseBuffer{stride, nullptr, data, false, allocated_by_malloc, size, use} {}

    virtual ~CPUBuffer() override {
        if (m_is_host_malloc) {
            free(m_data_host);
        }
    }

    virtual auto get_host_data() -> void * override {
        return m_data_host;
    }

    static auto create_buffer(Shape shape, size_t type_size) -> BufferPtr {
        Stride stride;
        stride[0] = type_size;
        for (size_t i = 1; i < shape.size(); i++) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        size_t size = stride.back() * shape.back();

        return std::make_shared<CPUBuffer>(stride, malloc(size), true, size, usage::COMPUTE);
    }

    static auto create_buffer_view(BaseBuffer &parent, Shape shape, size_t type_size, size_t offset = 0) -> BufferPtr {
        Stride stride;
        stride[0] = type_size;
        for (size_t i = 1; i < shape.size(); i++) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        auto &parent_buffer{static_cast<CPUBuffer &>(parent)};
        POWERSERVE_ASSERT(parent_buffer.m_data_host != nullptr, "parent buffer is nullptr");
        auto b    = std::make_shared<CPUBuffer>(stride, nullptr, false, parent_buffer.m_size, usage::COMPUTE);
        b->m_data_host = static_cast<void *>(static_cast<char*>(parent_buffer.m_data_host) + offset);
        return b;
    }
};

} // namespace powerserve::ggml
