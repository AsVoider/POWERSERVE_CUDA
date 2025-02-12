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

#include <memory>

namespace powerserve {

enum class usage : int {
    UNKNOWN = -1,
    ANY = 0,
    WEIGHT = 1,
    COMPUTE = 2,
};

struct BaseBuffer {
public:
    size_t m_size{0UL};
    usage  m_useage{usage::UNKNOWN};

public:
    virtual ~BaseBuffer() = default;
};

using BufferPtr = std::shared_ptr<BaseBuffer>;

} // namespace powerserve
