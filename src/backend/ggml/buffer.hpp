#pragma once

#include "core/tensor.hpp"

namespace smart::ggml {

struct Buffer : BaseBuffer {
    using Stride = std::array<size_t, Tensor::n_dims>;

    Stride stride;  // In bytes
    void *data;
};

}
