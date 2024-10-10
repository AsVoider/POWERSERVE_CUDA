#pragma once

#include "core/tensor.hpp"

namespace smart::ggml {

struct Buffer : BaseBuffer {
    using Stride = std::array<size_t, Tensor::max_n_dims>;

    Stride stride;  // In bytes
    void *data;
    bool allocated_by_malloc = false;

    Buffer(Stride stride_, void *data_, bool allocated_by_malloc_ = false) :
        stride(stride_),
        data(data_),
        allocated_by_malloc(allocated_by_malloc_)
    {}

    virtual ~Buffer() {
        if (allocated_by_malloc) {
            free(data);
        }
    }
};

}
