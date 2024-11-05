#pragma once

#include "common.hpp"
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
    int n_dims = 128;
    int n_ctx_orig = 2048;
    float freq_base = 10000.0f;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast = 32.0f;
    float beta_slow = 0.0f;
};

struct MHAParams {
    size_t layer_id = 0;
};

struct CopyParams {
    size_t off = 0;
};

struct PrintParams {
    size_t size = 0;
};

struct QuestAttnParams : OpParams {

    size_t layer_id_ = 0;
    std::vector<Region> &regions_;

    QuestAttnParams() = delete;

    explicit QuestAttnParams(size_t layer_id, std::vector<Region> &regions) : layer_id_(layer_id), regions_(regions) {}

    ~QuestAttnParams() override = default;
};

} // namespace smart
