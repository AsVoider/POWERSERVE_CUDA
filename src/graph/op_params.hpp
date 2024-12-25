#pragma once

#include "core/config.hpp"
#include "core/tensor.hpp"
#include "model/module/attention_mask.hpp"
#include "model/module/region.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

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

struct RMSNormParams {
    float eps;
};

struct RopeParams {
    std::vector<int> pos;
    ModelConfig::LLMConfig::RopeConfig rope_cfg;
};

struct AddCacheParams {
    size_t L;
    std::vector<int> pos;
    size_t head_id;
};

struct MHAParams {
    std::vector<int> pos;
    size_t layer_id  = 0;
    uint32_t n_heads = 0;
};

struct CopyParams {};

#if defined(SMART_WITH_QNN)
struct QNNForwardParams : OpParams {
    const CausalAttentionMask mask;
    std::vector<int> pos;

    explicit QNNForwardParams(std::vector<int> pos, const CausalAttentionMask &mask) : mask(mask), pos(pos) {}

    ~QNNForwardParams() override = default;
};

struct QNNForwardVLParams : OpParams {
    const CausalAttentionMask mask;
    std::vector<int> pos;

    std::vector<std::vector<float>> &pixel_values_list;
    std::vector<std::pair<int, size_t>> &img_infos;

    explicit QNNForwardVLParams(
        std::vector<int> pos,
        const CausalAttentionMask &mask,
        std::vector<std::vector<float>> &pixel_values_list,
        std::vector<std::pair<int, size_t>> &img_infos
    ) :
        mask(mask),
        pos(pos),
        pixel_values_list(pixel_values_list),
        img_infos(img_infos) {}

    ~QNNForwardVLParams() override = default;
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

struct PermuteParams {
    Tensor::Shape axes;
};

struct ContParams {};

struct ViewParams {
    Tensor::Shape stride;
    size_t offset;
};

struct SoftmaxExtParams {
    float scale;
    float max_bias;
};

struct GetMaskParams {
    const CausalAttentionMask &mask;
    const std::vector<int> &pos;
};

} // namespace smart
