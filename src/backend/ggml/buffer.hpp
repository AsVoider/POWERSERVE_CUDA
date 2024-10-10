#pragma once

#include "core/tensor.hpp"

namespace smart::ggml {

struct Buffer : BaseBuffer {
    using Stride = std::array<size_t, Tensor::n_dims>;

    Stride stride;  // In bytes
    void *data;

    Buffer(const Stride &stride_, void *data_) : stride(stride_), data(data_) {
        std::fill(std::begin(stride), std::end(stride), 1);

        SMART_ASSERT(stride.size() <= Tensor::n_dims);
        std::copy(stride_.begin(), stride_.end(), std::begin(stride));
 
    }

};

}
