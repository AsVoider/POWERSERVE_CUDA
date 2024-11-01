#include "sampler.hpp"

#include <random>

namespace smart {

void TemperatureMapper::apply(std::vector<ProbIndex> &probs) {
    if (m_temperature > 0) {
        // temperature scaling
        std::transform(probs.begin(), probs.end(), probs.begin(), [&](auto &y) {
            y.prob = y.prob / m_temperature;
            return y;
        });
    }
}

void SoftmaxMapper::apply(std::vector<ProbIndex> &probs) {
    softmax(probs);
}

void TopkSampler::apply(std::vector<ProbIndex> &probs) {
    if (m_topk <= 0) {
        return;
    }
    auto k = std::min(m_topk, probs.size());

    // Sort scores in descending order
    std::partial_sort(probs.begin(), probs.begin() + k, probs.end(), greater);
    probs.resize(k);
}

void ToppSampler::apply(std::vector<ProbIndex> &probs) {
    if (m_topp >= 1.0f) {
        return;
    }

    softmax(probs);

    // Compute the cumulative probabilities
    float cum_sum   = 0.0f;
    size_t last_idx = probs.size();

    for (size_t i = 0; i < probs.size(); ++i) {
        cum_sum += probs[i].prob;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= m_topp && i + 1 >= m_min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    probs.resize(last_idx);
}

void GreedySampler::apply(std::vector<ProbIndex> &probs) {
    auto max_prob = std::max_element(probs.begin(), probs.end(), less);
    probs[0]      = *max_prob;
    probs.resize(1);
}

struct probs_iterator {
    using iterator_category = std::input_iterator_tag;
    using value_type        = float;
    using pointer           = float *;
    using reference         = float &;
    using difference_type   = ptrdiff_t;

    const ProbIndex *data;

    probs_iterator(const ProbIndex *data) : data(data) {}

    bool operator==(const probs_iterator & other) const { return data == other.data; }
    bool operator!=(const probs_iterator & other) const { return !(*this == other); }
    const float & operator*() const { return data->prob; }
    probs_iterator & operator++() { ++data; return *this; }
    probs_iterator operator++(int) { probs_iterator tmp = *this; ++data; return tmp; }
};

void DistSampler::apply(std::vector<ProbIndex> &probs) {
    // SMART_UNUSED(probs);

    std::discrete_distribution<int> dist(probs_iterator{probs.data()}, probs_iterator{probs.data() + probs.size()});
    std::mt19937 gen(m_seed);
    auto idx = dist(gen);
    probs[0] = probs[idx];
    probs.resize(1);
}

void TemperatureExtMapper::apply(std::vector<ProbIndex> &probs) {
    if (m_delta > 0) {
        const float min_temp = std::max(0.0f, m_temperature - m_delta);
        const float max_temp = m_temperature + m_delta;
        float exponent_val = m_exponent;

        // no need to do anything if there is only one (or zero) candidates
        if (probs.size() <= 1) {
            return;
        }

        // Calculate maximum possible entropy
        float max_entropy = -logf(1.0f / probs.size());

        softmax(probs);

        // Calculate entropy of the softmax probabilities
        float entropy = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            float prob = probs[i].prob;
            if (prob > 0.0f) { // Ensure no log(0)
                entropy -= prob * logf(prob);
            }
        }

        // Normalize the entropy (max_entropy cannot be 0 here because we checked cur_p->size != 1 above)
        float normalized_entropy = entropy / max_entropy;

        // Map the normalized entropy to the desired temperature range using the power function
        float dyn_temp = min_temp + (max_temp - min_temp) * powf(normalized_entropy, exponent_val);

        {
            // fmt::println("Your text maxtemp value is: {}", max_temp);
            // fmt::println("Entropy: {}", entropy);
            // fmt::println("Max Possible Entropy: {}", max_entropy);
            // fmt::println("Normalized Entropy: {}", normalized_entropy);
            // fmt::println("Exponent: {}", exponent_val);
            // fmt::println("Dynamic Temperature (dyn_temp): {}", dyn_temp);
        }

        // Apply the dynamically calculated temperature scaling
        for (size_t i = 0; i < probs.size(); ++i) {
            probs[i].prob /= dyn_temp;
        }

        // Re-compute softmax probabilities after scaling logits with dynamic temperature
        const double max_l_double = probs[0].prob;

        double cum_sum_double = 0.0;
        for (size_t i = 0; i < probs.size(); ++i) {
            double p = exp(probs[i].prob - max_l_double);
            probs[i].prob = p; // Store the scaled probability
            cum_sum_double += p;
        }

        for (size_t i = 0; i < probs.size(); ++i) {
            probs[i].prob /= cum_sum_double; // Re-normalize the probabilities
        }

        {
            // Print the updated top 25 probabilities after temperature scaling
            // fmt::println("\nUpdated Top 25 Probabilities After Dynamic Temperature Scaling (in percentages):");
            // for (size_t i = 0; i < 25 && i < probs.size(); ++i) {
            //     fmt::println("Token {}: {}%", i + 1, probs[i].prob * 100.0f);
            // }
        }

    } else {
        for (size_t i = 0; i < probs.size(); ++i) {
            probs[i].prob /= m_temperature;
        }
    }
}

} // namespace smart
