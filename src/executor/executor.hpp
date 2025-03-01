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

#include "backend/platform.hpp"
#include "graph/graph.hpp"
#include "graph/graph_split.hpp"

namespace powerserve {

struct Executor {
public:
    Platform &m_platform;
    Graph &m_graph;
    std::vector<std::unique_ptr<GraphSplit>> graph_splits{};

public:
    Executor(Platform &platform, Graph &graph) : m_platform(platform), m_graph(graph) {}

public:
    void shed_op_to_backend();
    void allocate_buffer_with_backend();
    void print_graph(std::ostream &os);
    void split_graph();
    void run_with_backend();

private:
    template <typename T>
    void create_backend_buffer(std::shared_ptr<TensorNode> tensor) {
        if (tensor->type == NodeType::TENSOR_VIEW) {
            tensor->m_data = Platform::buffer_interfaces.at(tensor->m_backend).create_buffer_view(
                *tensor->tensor_view()->parent->m_data, tensor->m_shape, sizeof(T), 0UL
            );
        } else {
            tensor->m_data = Platform::buffer_interfaces.at(tensor->m_backend).create_buffer(tensor->m_shape, sizeof(T));
        }
    }
};

} // namespace powerserve
