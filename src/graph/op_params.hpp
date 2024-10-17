#pragma once

#include <cstddef>

namespace smart {

// Base class for op parameters
struct OpParams {
    virtual ~OpParams() = default;
};

// This wrapper decouples the inheritance from the parameter structs
// So that parameter structs can keep its default constructors
template <typename T>
struct OpParamWrapper : OpParams {
    T value;

    explicit OpParamWrapper(const T &value) : value(value) {}
};

struct MHAParams {
    size_t layer_id = 0;
};

struct CopyParams {
    size_t off = 0;
};

} // namespace smart
