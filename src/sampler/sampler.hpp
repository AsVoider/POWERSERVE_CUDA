#pragma once

#include "common.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

namespace smart {

struct ProbIndex {
    float prob;
    int index;
};

const auto less    = [](const ProbIndex &a, const ProbIndex &b) { return a.prob < b.prob; };
const auto greater = [](const ProbIndex &a, const ProbIndex &b) { return a.prob > b.prob; };

static void softmax(std::vector<ProbIndex> &probs) {
    auto max_val{std::max_element(probs.begin(), probs.end(), less)->prob};

    // exp and sum
    std::transform(probs.begin(), probs.end(), probs.begin(), [&](auto &y) {
        y.prob = expf(y.prob - max_val);
        return y;
    });
    double sum{std::accumulate(probs.begin(), probs.end(), 0.0f, [](float acc, const ProbIndex &b) {
        return acc + b.prob;
    })};

    // normalize
    std::transform(probs.begin(), probs.end(), probs.begin(), [&](auto &y) {
        y.prob = y.prob / sum;
        return y;
    });
}

static constexpr uint32_t DEFAULT_SEED = 0;

static uint32_t get_rng_seed(uint32_t seed) {
    if (seed == DEFAULT_SEED) {
        // use system clock if std::random_device is not a true RNG
        static bool is_rd_prng = std::random_device().entropy() == 0;
        if (is_rd_prng) {
            return (uint32_t)std::chrono::system_clock::now().time_since_epoch().count();
        }
        std::random_device rd;
        return rd();
    }
    return seed;
}

struct Sampler {
public:
    Sampler()          = default;
    virtual ~Sampler() = default;

public:
    virtual int sample(std::vector<float> &logits) {
        SMART_UNUSED(logits);
        return 0;
    }

    virtual void apply(std::vector<ProbIndex> &probs) = 0;
};

// Mapper must not resize the probs size
struct TemperatureMapper : Sampler {
public:
    float m_temperature = 0.6f;

public:
    TemperatureMapper(float temperature) : m_temperature(temperature) {}

    ~TemperatureMapper() = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
};

struct SoftmaxMapper : Sampler {
public:
    SoftmaxMapper()  = default;
    ~SoftmaxMapper() = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
};

struct TopkSampler : Sampler {
public:
    size_t m_topk = 40;

public:
    TopkSampler(size_t topk) : m_topk(topk) {}

    ~TopkSampler() = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
};

struct ToppSampler : Sampler {
public:
    float m_topp      = 0.95f;
    size_t m_min_keep = 1; // what's this

public:
    ToppSampler(float topp, size_t min_keep = 1) : m_topp(topp), m_min_keep(min_keep) {}

    ~ToppSampler() = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
};

struct GreedySampler : Sampler {
public:
    ~GreedySampler() = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
};

struct DistSampler : Sampler {
public:
    uint64_t m_seed = 0;

public:
    DistSampler(uint64_t seed) : m_seed(get_rng_seed(seed)) {}

    ~DistSampler() = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
};

struct TemperatureExtMapper : Sampler {
public:
    float m_temperature = 0.6f;
    float m_delta       = 0.00f; // 0.0 = disabled
    float m_exponent    = 1.00f; // controls how entropy maps to temperature in dynamic temperature sampler

public:
    TemperatureExtMapper(float temperature, float delta = 0.0f, float exponent = 1.0f) :
        m_temperature(temperature),
        m_delta(delta),
        m_exponent(exponent) {}

    ~TemperatureExtMapper() = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
};

} // namespace smart
