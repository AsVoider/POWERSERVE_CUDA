#pragma once

#include <vector>

namespace smart {

struct ProbIndex {
    float prob;
    int index;
};

struct Sampler {
public:
    Sampler()          = default;
    virtual ~Sampler() = default;

public:
    virtual int sample(std::vector<float> &logits) = 0;
    virtual void trans(std::vector<float> &logits) = 0;
};

} // namespace smart
