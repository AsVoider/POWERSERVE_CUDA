#pragma once

#include "common.hpp"
#include "core/config.hpp"
#include "model/module/attention_mask.hpp"
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

struct GetEmbeddingParams {
    std::vector<int> tokens;
};

struct RopeParams {
    std::vector<int> pos;
    RopeConfig rope_cfg;
};

struct AddCacheParams {
    size_t L;
    std::vector<int> pos;
    size_t head_id;
    bool is_k;
};

struct MHAParams {
    std::vector<int> pos;
    size_t layer_id  = 0;
    uint32_t n_heads = 0;
};

struct CopyParams {
    size_t off = 0;
};

#if defined(SMART_WITH_QNN)
struct QNNForwardParams : OpParams {
    const CausalAttentionMask mask;
    std::vector<int> pos;

    explicit QNNForwardParams(std::vector<int> pos, const CausalAttentionMask &mask) : mask(mask), pos(pos) {}

    ~QNNForwardParams() override = default;
};
#endif

struct PrintParams {
    size_t size = 0;
};

struct QuestAttnParams {
    std::vector<int> pos;
    size_t layer_id = 0;
    std::vector<Region> &regions;
    uint32_t n_heads = 0;
};

} // namespace smart
