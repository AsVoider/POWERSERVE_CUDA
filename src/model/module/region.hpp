#pragma once

#include "common.hpp"

#include <cstdint>
#include <vector>

namespace smart {

constexpr int64_t REGION_TOKENS = 4;

struct Region {

    struct RegionTensor {
        uint32_t L;
        uint32_t pos;
    };

    std::vector<RegionTensor> region_tensors;
    int64_t tokens_limit;
    uint32_t kv_dim;

    // quest relative
    std::vector<float> key_max_scores; // (kv_dim,)
    std::vector<float> key_min_scores; // (kv_dim,)

    // final score
    float final_score;

    Region() = default;

    Region(uint32_t kv_dim, int64_t tokens_limit = REGION_TOKENS) :
        region_tensors(),
        tokens_limit(tokens_limit),
        kv_dim(kv_dim),
        final_score(0) {
        key_max_scores.resize(kv_dim, -INFINITY);
        key_min_scores.resize(kv_dim, INFINITY);
    }

    ~Region() = default;

    float score(float *query, uint32_t kv_mul) {
        float score = 0;
        for (uint32_t i = 0; i < kv_dim; i += kv_mul) {
            for (uint32_t j = 0; j < kv_mul; ++j) { // query (dim, )
                score += std::max(key_max_scores[i] * query[i + j * kv_dim], key_min_scores[i] * query[i + j * kv_dim]);
            }
        }
        final_score = score;
        return score;
    }

    void update_score(float *key_cache, uint32_t seq_len, uint32_t L, uint32_t pos) {
        SMART_ASSERT((int64_t)region_tensors.size() < tokens_limit);
        region_tensors.emplace_back(L, pos);
        auto new_score = key_cache + L * seq_len * kv_dim + pos * kv_dim;

        for (uint32_t i = 0; i < kv_dim; ++i) {
            key_max_scores[i] = std::max(key_max_scores[i], new_score[i]);
            key_min_scores[i] = std::min(key_min_scores[i], new_score[i]);
        }
    }

    bool is_full() {
        return (int64_t)region_tensors.size() >= tokens_limit;
    }
};

} // namespace smart