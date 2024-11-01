#pragma once

#include "sampler.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace smart {

struct SamplerConfig {
    uint64_t seed   = 0;
    float temp      = 0.80f;
    float topp      = 0.95f; // 1.0 = disabled
    size_t topk     = 40;
    size_t min_keep = 0; // 0 = disabled, otherwise samplers should return at least min_keep tokens
};

struct SamplerChain : Sampler {
private:
    std::vector<ProbIndex> m_probs{};

public:
    SamplerConfig m_config{};
    std::vector<std::unique_ptr<Sampler>> m_samplers{};

public:
    SamplerChain(SamplerConfig config);

    ~SamplerChain() = default;

public:
    int sample(std::vector<float> &logits) override;
    void apply(std::vector<ProbIndex> &probs) override;

private:
    void convert_logits(const std::vector<float> &logits);
};

} // namespace smart