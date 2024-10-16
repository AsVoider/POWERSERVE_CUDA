#pragma once

#include <vector>

namespace smart {

struct Sampler {
public:
    Sampler()          = default;
    virtual ~Sampler() = default;

public:
    virtual int sample(std::vector<float> &logits) = 0;
};

} // namespace smart
