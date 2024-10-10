#pragma once

#include "common.hpp"

#include <memory>

namespace smart {

struct Tensor;

struct BaseBuffer {
    virtual ~BaseBuffer() = default;
};

using BufferPtr = std::shared_ptr<BaseBuffer>;

}
