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
    void plan();

private:
    template <typename T>
    void create_cpu_buffer(std::shared_ptr<TensorNode> tensor) {
        if (tensor->type == NodeType::TENSOR_VIEW) {
            tensor->m_data =
                CPUBuffer::create_buffer_view<T>(tensor->tensor_view()->parent->get<CPUBuffer>(), tensor->m_shape);
        } else {
            tensor->m_data = CPUBuffer::create_buffer<T>(tensor->m_shape);
        }
    }
};

} // namespace smart
