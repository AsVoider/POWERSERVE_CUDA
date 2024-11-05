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

struct ProbArray {
public:
    std::vector<ProbIndex> m_probs;
    bool m_is_sorted     = false;
    bool m_is_normalized = false;

public:
    ProbArray(std::vector<float> &logits) {
        m_probs.resize(logits.size());
        for (size_t i = 0; i < logits.size(); i++) {
            m_probs[i].index = i;
            m_probs[i].prob  = logits[i];
        }
    }

private:
    struct probs_iterator {
        using iterator_category = std::input_iterator_tag;
        using value_type        = float;
        using pointer           = float *;
        using reference         = float &;
        using difference_type   = ptrdiff_t;

        const ProbIndex *data;

        probs_iterator(const ProbIndex *data) : data(data) {}

        bool operator==(const probs_iterator &other) const {
            return data == other.data;
        }

        bool operator!=(const probs_iterator &other) const {
            return !(*this == other);
        }

        const float &operator*() const {
            return data->prob;
        }

        probs_iterator &operator++() {
            ++data;
            return *this;
        }

        probs_iterator operator++(int) {
            probs_iterator tmp = *this;
            ++data;
            return tmp;
        }
    };

public:
    template <typename RandomEngine>
    ProbIndex sample(RandomEngine &&gen) {
        std::discrete_distribution<int> dist(
            probs_iterator{m_probs.data()}, probs_iterator{m_probs.data() + m_probs.size()}
        );
        auto idx = dist(gen);
        return m_probs[idx];
    }
};

static void normalize(ProbArray &probs) {
    double sum = 0.;
    for (const auto &p : probs.m_probs) {
        sum += p.prob;
    }

    // normalize
    for (auto &p : probs.m_probs) {
        p.prob /= sum;
    }
    probs.m_is_normalized = true;
}

static void softmax(ProbArray &probs) {
    SMART_ASSERT(probs.m_probs.size() > 0);
    if (!probs.m_is_sorted) {
        std::sort(probs.m_probs.begin(), probs.m_probs.end(), std::greater<>());
        probs.m_is_sorted = true;
    }

    auto max_val = probs.m_probs[0].prob;

    // exp
    for (auto &p : probs.m_probs) {
        p.prob = std::exp(p.prob - max_val);
    }

    // normalize
    normalize(probs);
}

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
    Sampler()          = default;
    virtual ~Sampler() = default;

public:
    virtual void apply(ProbArray &probs) = 0;

    virtual void accept(Tokenizer::Token token) {
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

// struct StochasticSampler : Sampler {
// public:
//     uint64_t m_seed = 0;

// public:
//     StochasticSampler(uint64_t seed) : m_seed(get_rng_seed(seed)) {}

//     virtual ~StochasticSampler() override = default;

// public:
//     void apply(ProbArray &probs) override;
// };

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
    void accept(Tokenizer::Token token) override;
};

} // namespace smart
