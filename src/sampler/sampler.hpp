#pragma once

#include <vector>
namespace smart {

class Sampler{
public:
    virtual int sample(std::vector<float> &logits) = 0;
    Sampler() = default;
    ~Sampler() = default;
};

} // namespace smart