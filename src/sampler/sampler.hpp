#pragma once

#include "common/logger.hpp"
#include "common/type_def.hpp"

#include <cmath>
#include <cstddef>
#include <deque>
#include <random>
#include <vector>

namespace smart {

struct ProbIndex {
    float prob  = 0.0f;
    Token index = -1;

    bool operator<(const ProbIndex &other) const {
        return prob < other.prob;
    }

    bool operator>(const ProbIndex &other) const {
        return prob > other.prob;
    }
};

struct ProbArray {
public:
    std::vector<ProbIndex> m_probs;
    bool m_is_sorted     = false;
    bool m_is_normalized = false;

public:
    ProbArray(const std::vector<float> &logits) {
        m_probs.resize(logits.size());
        for (size_t i = 0; i < logits.size(); i++) {
            m_probs[i].index = i;
            m_probs[i].prob  = logits[i];
        }
    }

public:
    template <typename RandomEngine>
    ProbIndex sample(RandomEngine &&gen) {
        size_t index = std::discrete_distribution(m_probs.size(), 0, m_probs.size(), [&](double x) {
            // https://en.cppreference.com/w/cpp/numeric/random/discrete_distribution/discrete_distribution
            // x = i + 0.5
            return m_probs[(size_t)x].prob;
        })(gen);

        return m_probs[index];
    }

public:
    void normalize();
    void softmax();
};

static constexpr uint64_t DEFAULT_SEED = uint64_t(-1);

static uint64_t get_rng_seed(uint64_t seed) {
    if (seed == DEFAULT_SEED) {
        // use system clock if std::random_device is not a true RNG
        // static bool is_rd_prng = std::random_device().entropy() == 0;
        // if (is_rd_prng) {
        //     return (uint32_t)std::chrono::system_clock::now().time_since_epoch().count();
        // }
        std::random_device rd;
        return rd();
    }
    return seed;
}

struct Sampler {
public:
    virtual ~Sampler() = default;

public:
    virtual void apply(ProbArray &probs) = 0;

    virtual void accept(Token token) {
        SMART_UNUSED(token);
    }
};

// Mapper must not resize the probs size
struct TemperatureSampler : Sampler {
public:
    float m_temperature = 0.6f;

public:
    TemperatureSampler(float temperature) : m_temperature(temperature) {}

    virtual ~TemperatureSampler() override = default;

public:
    void apply(ProbArray &probs) override;
};

struct SoftmaxSampler : Sampler {
public:
    virtual ~SoftmaxSampler() override = default;

public:
    void apply(ProbArray &probs) override;
};

struct NormalizeSampler : Sampler {
public:
    virtual ~NormalizeSampler() override = default;

public:
    void apply(ProbArray &probs) override;
};

struct TopKSampler : Sampler {
public:
    size_t m_topk = 40;

public:
    TopKSampler(size_t topk) : m_topk(topk) {}

    virtual ~TopKSampler() override = default;

public:
    void apply(ProbArray &probs) override;
};

struct TopPSampler : Sampler {
public:
    float m_topp      = 0.95f;
    size_t m_min_keep = 1; // what's this

public:
    TopPSampler(float topp, size_t min_keep = 1) : m_topp(topp), m_min_keep(min_keep) {}

    virtual ~TopPSampler() override = default;

public:
    void apply(ProbArray &probs) override;
};

struct GreedySampler : TopKSampler {
public:
    size_t m_topk = 1;

public:
    GreedySampler() : TopKSampler(1) {}

    virtual ~GreedySampler() override = default;
};

struct TemperatureExtSampler : Sampler {
public:
    float m_temperature = 0.6f;
    float m_delta       = 0.00f; // 0.0 = disabled
    float m_exponent    = 1.00f; // controls how entropy maps to temperature in dynamic temperature sampler

public:
    TemperatureExtSampler(float temperature, float delta = 0.0f, float exponent = 1.0f) :
        m_temperature(temperature),
        m_delta(delta),
        m_exponent(exponent) {}

    virtual ~TemperatureExtSampler() override = default;

public:
    void apply(ProbArray &probs) override;
};

constexpr auto NULL_TOKEN = -1;

struct PenaltyChecker : Sampler {
public:
    int32_t m_vocab_size;
    Token m_special_eos_id;
    Token m_linefeed_id;

    int32_t m_penalty_last_n;
    float m_penalty_repeat;
    float m_penalty_freq;
    float m_penalty_present;

    bool m_penalize_nl;
    bool m_ignore_eos;

    std::deque<Token> m_prev;

public:
    PenaltyChecker(
        int32_t vocab_size,
        Token special_eos_id,
        Token linefeed_id,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        bool penalize_nl,
        bool ignore_eos
    ) :
        m_vocab_size(vocab_size),
        m_special_eos_id(special_eos_id),
        m_linefeed_id(linefeed_id),
        m_penalty_last_n(penalty_last_n),
        m_penalty_repeat(penalty_repeat),
        m_penalty_freq(penalty_freq),
        m_penalty_present(penalty_present),
        m_penalize_nl(penalize_nl),
        m_ignore_eos(ignore_eos) {
        if (linefeed_id == NULL_TOKEN) {
            m_penalize_nl = true;
        }

        if (special_eos_id == NULL_TOKEN) {
            m_ignore_eos = false;
        }
        penalty_last_n = std::max(penalty_last_n, 0);
        m_prev.resize(m_penalty_last_n);
    }

    virtual ~PenaltyChecker() override = default;

public:
    void apply(ProbArray &probs) override;
    void accept(Token token) override;
};

} // namespace smart
