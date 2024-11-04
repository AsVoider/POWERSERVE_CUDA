#pragma once

#include "common.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cmath>
#include <cstddef>
#include <deque>
#include <random>
#include <vector>

namespace smart {

struct ProbIndex {
    float prob;
    Tokenizer::Token index;

    bool operator<(const ProbIndex &other) const {
        return prob < other.prob;
    }

    bool operator>(const ProbIndex &other) const {
        return prob > other.prob;
    }
};

static void softmax(std::vector<ProbIndex> &probs) {
    auto max_val{std::max_element(probs.begin(), probs.end())->prob};

    // exp and sum
    for (auto &p : probs) {
        p.prob = std::exp(p.prob - max_val);
    }
    // auto sum{std::reduce(probs.begin(), probs.end(), ProbIndex{0.0f, 0}, [](const ProbIndex &a, const ProbIndex &b) {
    //     return ProbIndex{a.prob + b.prob, 0};
    // }).prob};
    double sum = 0.;
    for (const auto &p : probs) {
        sum += p.prob;
    }

    // normalize
    for (auto &p : probs) {
        p.prob /= sum;
    }
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
    virtual void apply(std::vector<ProbIndex> &probs) = 0;
};

// Mapper must not resize the probs size
struct TemperatureMapper : Sampler {
public:
    float m_temperature = 0.6f;

public:
    TemperatureMapper(float temperature) : m_temperature(temperature) {}

    virtual ~TemperatureMapper() override = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
};

struct SoftmaxMapper : Sampler {
public:
    SoftmaxMapper()                   = default;
    virtual ~SoftmaxMapper() override = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
};

struct TopKSampler : Sampler {
public:
    size_t m_topk = 40;

public:
    TopKSampler(size_t topk) : m_topk(topk) {}

    virtual ~TopKSampler() override = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
};

struct TopPSampler : Sampler {
public:
    float m_topp      = 0.95f;
    size_t m_min_keep = 1; // what's this

public:
    TopPSampler(float topp, size_t min_keep = 1) : m_topp(topp), m_min_keep(min_keep) {}

    virtual ~TopPSampler() override = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
};

// struct GreedySampler : Sampler {
// public:
//     ~GreedySampler() = default;

// public:
//     void apply(std::vector<ProbIndex> &probs) override;
// };

struct StochasticSampler : Sampler {
public:
    uint64_t m_seed = 0;

public:
    StochasticSampler(uint64_t seed) : m_seed(get_rng_seed(seed)) {}

    virtual ~StochasticSampler() override = default;

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

    virtual ~TemperatureExtMapper() override = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
};

constexpr auto Token_NULL = -1;

struct PenaltyChecker : Sampler {

public:
    int32_t m_vocab_size;
    Tokenizer::Token m_special_eos_id;
    Tokenizer::Token m_linefeed_id;

    int32_t m_penalty_last_n;
    float m_penalty_repeat;
    float m_penalty_freq;
    float m_penalty_present;

    bool m_penalize_nl;
    bool m_ignore_eos;

    std::deque<Tokenizer::Token> m_prev;

public:
    PenaltyChecker(
        int32_t vocab_size,
        Tokenizer::Token special_eos_id,
        Tokenizer::Token linefeed_id,
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
        if (linefeed_id == Token_NULL) {
            m_penalize_nl = true;
        }

        if (special_eos_id == Token_NULL) {
            m_ignore_eos = false;
        }
        penalty_last_n = std::max(penalty_last_n, 0);
        m_prev.resize(m_penalty_last_n);
    }

    virtual ~PenaltyChecker() override = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
    void accept(Tokenizer::Token token);
};

} // namespace smart
