#include "greedy_sampler.hpp"

#include <algorithm>
#include <iterator>

namespace smart {

int GreedySampler::sample(std::vector<float> &logits) {
    auto max_it = std::max_element(logits.begin(), logits.end());
    return std::distance(logits.begin(), max_it);
}

} // namespace smart
