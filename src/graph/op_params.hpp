#pragma once

#include "common.hpp"
#include "core/config.hpp"
#include "model/module/region.hpp"

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

struct RopeParams {
    RopeConfig rope_cfg;
};

struct MHAParams {
    size_t layer_id  = 0;
    uint32_t n_heads = 0;
};

struct CopyParams {
    size_t off = 0;
};

struct PrintParams {
    size_t size = 0;
};

struct QuestAttnParams {
    size_t layer_id = 0;
    std::vector<Region> &regions;
    uint32_t n_heads = 0;
};

} // namespace smart
