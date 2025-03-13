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

#include "typedefs.hpp"

#include <memory>

namespace powerserve {

enum class usage : int64_t {
    UNKNOWN = -1,
    ANY     = 0,
    WEIGHT  = 1,
    COMPUTE = 2,
};

struct BaseBuffer {
public:
    Stride m_stride;
    void *m_data_device{nullptr};
    void *m_data_host{nullptr};
    bool m_is_device_malloc{false};
    bool m_is_host_malloc{false};
    size_t m_size{0UL};
    usage m_useage{usage::UNKNOWN};

public:
    BaseBuffer(
        Stride stride, void *data_device, void *data_host, bool device_malloc, bool host_malloc, size_t size, usage use
    ) :
        m_stride{stride},
        m_data_device{data_device},
        m_data_host{data_host},
        m_is_device_malloc{device_malloc},
        m_is_host_malloc{host_malloc},
        m_size{size},
        m_useage{use} {}

    virtual ~BaseBuffer()                  = default;
    virtual auto get_host_data() -> void * = 0;
};

using BufferPtr = std::shared_ptr<BaseBuffer>;

} // namespace powerserve
