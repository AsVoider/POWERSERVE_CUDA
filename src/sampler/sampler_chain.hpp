#pragma once

#include "sampler.hpp"

#include <cstddef>
#include <numeric>
#include <vector>
#include <algorithm>
#include <cmath>

namespace smart {

const auto less = [](const ProbIndex &a, const ProbIndex &b) { return a.prob < b.prob; };
const auto greater = [](const ProbIndex &a, const ProbIndex &b) { return a.prob > b.prob; };

static void softmax(std::vector<ProbIndex> &logits) {
    // find max value (for numerical stability)
    auto max_val{std::max_element(logits.begin(), logits.end(), less)->prob};

    // exp and sum
    std::transform(logits.begin(), logits.end(), logits.begin(), [&](auto &y) { y.prob = expf(y.prob - max_val); return y;});
    double sum{std::accumulate(logits.begin(), logits.end(), 0.0f, [](float acc, const ProbIndex &b) { return acc + b.prob; })};

    // normalize
    std::transform(logits.begin(), logits.end(), logits.begin(), [&](auto &y) { y.prob = y.prob / sum; return y; });
}

static std::vector<ProbIndex> convert_logits(const std::vector<float> &logits) {
    std::vector<ProbIndex> probindex(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        probindex[i].index = i;
        probindex[i].prob = logits[i];
    }
    return probindex;
}

static void apply_temperature(std::vector<ProbIndex> &logits, float temperature) {
    if (temperature > 0) {
        // temperature scaling
        std::transform(logits.begin(), logits.end(), logits.begin(), [&](auto &y) { y.prob = y.prob / temperature; return y; });
        // softmax
        softmax(logits);
    }
}

static void topk_sample(std::vector<ProbIndex> &logits, size_t k) {
    if (k <= 0) {
        return;
    }

    k = std::min(k, logits.size());
    // Sort scores in descending order
    std::partial_sort(logits.begin(), logits.begin() + k, logits.end(), greater);
    logits.resize(k);
}

static ProbIndex greedy_sample(std::vector<ProbIndex> &logits) {
    // auto max_it = std::max_element(logits.begin(), logits.end(), less);
    // return *max_it;
    topk_sample(logits, 1);
    return logits[0];
}

} // namespace smart