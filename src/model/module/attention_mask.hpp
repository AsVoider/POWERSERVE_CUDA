#pragma once

#include "common.hpp"

namespace smart {

struct AttentionMask {
    size_t size = 0;

    AttentionMask(size_t size);

    virtual ~AttentionMask() = default;

    virtual bool not_masked(size_t i, size_t j) const = 0;
};

struct AttentionMaskView {
    const AttentionMask &mask;
    size_t offset = 0;

    size_t size = 0;

    AttentionMaskView(const AttentionMask &mask, size_t offset, size_t size);

    bool not_masked(size_t i, size_t j) const;
};

struct CausalAttentionMask : AttentionMask {
    std::vector<std::vector<bool>> mask;

    CausalAttentionMask(size_t size);
    CausalAttentionMask(size_t size, const std::vector<std::vector<bool>> &mask);
    virtual ~CausalAttentionMask() override = default;
    virtual bool not_masked(size_t i, size_t j) const override;
};

} // namespace smart
