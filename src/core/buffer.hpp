#pragma once

#include "common.hpp"

namespace smart {

struct Tensor;

struct BaseBuffer {
    virtual ~BaseBuffer() = default;
};

}
