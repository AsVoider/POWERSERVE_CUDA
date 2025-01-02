#pragma once

#include "common/logger.hpp"
#include "common/type_def.hpp"
#include "sampler/prob_array.hpp"

#include <cstddef>
#include <deque>

namespace smart {

struct Sampler {
    virtual ~Sampler() = default;

    virtual void apply(ProbArray &probs) = 0;

    virtual void accept(Token token) {
        SMART_UNUSED(token);
    }
};

struct TemperatureSampler final : Sampler {
    float m_temperature = 0.6f;

    TemperatureSampler(float temperature) : m_temperature(temperature) {}

    void apply(ProbArray &probs) override;
};

struct SoftmaxSampler final : Sampler {
    void apply(ProbArray &probs) override;
};

struct NormalizeSampler final : Sampler {
    void apply(ProbArray &probs) override;
};

struct TopKSampler final : Sampler {
    size_t m_topk = 40;

    TopKSampler(size_t topk) : m_topk(topk) {}

    void apply(ProbArray &probs) override;
};

struct TopPSampler final : Sampler {
    float m_topp      = 0.95f;
    size_t m_min_keep = 1; // what's this

    TopPSampler(float topp, size_t min_keep = 1) : m_topp(topp), m_min_keep(min_keep) {}

    void apply(ProbArray &probs) override;
};

struct RepeatPenaltySampler final : Sampler {
    static constexpr Token null_token = -1;

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

    RepeatPenaltySampler(
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
        if (linefeed_id == null_token) {
            m_penalize_nl = true;
        }

        if (special_eos_id == null_token) {
            m_ignore_eos = false;
        }
        penalty_last_n = std::max(penalty_last_n, 0);
        m_prev.resize(m_penalty_last_n);
    }

    void apply(ProbArray &probs) override;
    void accept(Token token) override;
};

struct StochasticSampler final : Sampler {
    std::mt19937 m_random_state;

    StochasticSampler(uint64_t seed);

    void apply(ProbArray &probs) override;
};

} // namespace smart
