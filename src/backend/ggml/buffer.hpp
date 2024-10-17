#pragma once

#include "core/tensor.hpp"

#include <cstddef>

namespace smart::ggml {

struct Buffer : BaseBuffer {
public:
    using Stride = std::array<size_t, Tensor::max_n_dims>;

public:
    Stride m_stride; // In bytes
    void *m_data;
    bool m_allocated_by_malloc = false;

public:
    Buffer(Stride stride, void *data, bool allocated_by_malloc = false) :
        m_stride(stride),
        m_data(data),
        m_allocated_by_malloc(allocated_by_malloc) {}

    virtual ~Buffer() override {
        if (m_allocated_by_malloc) {
            free(m_data);
        }
    }
};

} // namespace smart::ggml
