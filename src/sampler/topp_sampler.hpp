#pragma once

#include "sampler.hpp"

#include <cstddef>
#include <stdint.h>
#include <vector>

namespace smart {

struct ToppSampler : Sampler {
public:
    float m_temperature  = 0.6f;
    float m_topp         = 0.9f;
    uint64_t m_rng_state = 0;

public:
    ToppSampler() = default;

    ToppSampler(float temperature, float topp, uint64_t rng_state) :
        m_temperature(temperature),
        m_topp(topp),
        m_rng_state(rng_state) {}

    ~ToppSampler() override = default;

public:
    int sample(std::vector<float> &logits) override;
    void trans(std::vector<float> &logits) override;
};

} // namespace smart
