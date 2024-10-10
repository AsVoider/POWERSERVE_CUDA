#pragma once

#include "common.hpp"

namespace smart {

struct Tensor;

struct BaseBuffer {
    Tensor *tensor = nullptr;
    size_t n_bytes = 0;

    virtual ~BaseBuffer() = default;
};

}
