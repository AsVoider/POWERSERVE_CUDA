#include "greedy_sampler.hpp"

#include "common.hpp"

#include <algorithm>
#include <iterator>

namespace smart {

int GreedySampler::sample(std::vector<float> &logits) {
    auto max_it = std::max_element(logits.begin(), logits.end());
    return std::distance(logits.begin(), max_it);
}

void GreedySampler::trans(std::vector<float> &logits) {
    auto idx    = sample(logits);
    auto target = logits[idx];
    // precesion in < 1e-6f
    auto func = [&](float x) { return x < target ? -INFINITY : x; };
    std::transform(logits.begin(), logits.end(), logits.begin(), func);
}

} // namespace smart
