#include "attention_mask.hpp"

namespace smart {

AttentionMask::AttentionMask(size_t size) : size(size) {}

AttentionMaskView::AttentionMaskView(const AttentionMask &mask, size_t offset, size_t size) :
    mask(mask),
    offset(offset),
    size(size) {}

bool AttentionMaskView::not_masked(size_t i, size_t j) const {
    return mask.not_masked(offset + i, offset + j);
}

CausalAttentionMask::CausalAttentionMask(size_t size) : AttentionMask(size) {}

CausalAttentionMask::CausalAttentionMask(size_t size, const std::vector<std::vector<bool>> &batch_mask) :
    AttentionMask(size),
    mask(batch_mask) {}

bool CausalAttentionMask::not_masked(size_t i, size_t j) const {
    if (!mask.empty()) {
        return mask[i][j];
    }
    return i >= j;
}

} // namespace smart
