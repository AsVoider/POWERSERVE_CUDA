#pragma once

#include "backend/platform.hpp"
#include "graph/graph.hpp"

namespace smart {

struct Executor {
public:
    Platform &m_platform;
    Graph &m_graph;

public:
    Executor(Platform &platform, Graph &graph) : m_platform(platform), m_graph(graph) {}

public:
    void allocate_buffers();
    void run();

private:
    template <typename T>
    void create_ggml_buffer(std::shared_ptr<TensorNode> tensor) {
        if (tensor->type == NodeType::TENSOR_VIEW) {
            tensor->m_data = m_platform.ggml_backend->create_buffer_view<T>(
                tensor->tensor_view()->parent->get<ggml::Buffer>(), tensor->m_shape
            );
        } else {
            tensor->m_data = m_platform.ggml_backend->create_buffer<T>(tensor->m_shape);
        }
    }
};

} // namespace smart
