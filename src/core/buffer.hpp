#pragma once

#include <memory>

namespace smart {

struct Tensor;

struct BaseBuffer {
public:
    virtual ~BaseBuffer() = default;
};

using BufferPtr = std::shared_ptr<BaseBuffer>;

} // namespace smart
