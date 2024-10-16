#pragma once

#include "sampler.hpp"

#include <vector>

namespace smart {

struct GreedySampler : Sampler {
public:
    GreedySampler()           = default;
    ~GreedySampler() override = default;

public:
    int sample(std::vector<float> &logits) override;
};

} // namespace smart
