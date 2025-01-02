#pragma once

#include "common/logger.hpp"
#include "common/type_def.hpp"

#include <random>

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
    std::vector<ProbIndex> m_probs;
    bool m_is_sorted     = false; // Is sorted in descending order?
    bool m_is_normalized = false; // Does the sum of probs equal to 1?

    ProbArray(const std::vector<float> &logits) {
        m_probs.resize(logits.size());
        for (size_t i = 0; i < logits.size(); i++) {
            m_probs[i].index = i;
            m_probs[i].prob  = logits[i];
        }
    }

    auto operator[](size_t i) -> ProbIndex & {
        return m_probs[i];
    }

    void normalize();
    void softmax();

    void resize(size_t size);

    template <typename RandomEngine>
    auto stochastic_sample(RandomEngine &&gen) -> ProbIndex & {
        SMART_ASSERT(m_is_normalized);

        size_t index = std::discrete_distribution(m_probs.size(), 0, m_probs.size(), [&](double x) {
            // https://en.cppreference.com/w/cpp/numeric/random/discrete_distribution/discrete_distribution
            // x = i + 0.5
            return m_probs[(size_t)x].prob;
        })(gen);

        return m_probs[index];
    }

    auto greedy_sample() -> ProbIndex &;
};

} // namespace smart
