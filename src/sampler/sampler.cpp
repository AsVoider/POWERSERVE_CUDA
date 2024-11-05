#include "sampler.hpp"

#include <random>

namespace smart {

ProbIndex ProbArray::sample() {
    return m_probs[0];
}

void TemperatureSampler::apply(ProbArray &probs) {
    if (m_temperature > 0) {
        // temperature scaling
        for (auto &prob : probs.m_probs) {
            prob.prob /= m_temperature;
        }
    }
}

void SoftmaxSampler::apply(ProbArray &probs) {
    softmax(probs);
}

void NormalizeSampler::apply(ProbArray &probs) {
    normalize(probs);
}

void TopKSampler::apply(ProbArray &probs) {
    SMART_ASSERT(m_topk > 0);
    auto k = std::min(m_topk, probs.m_probs.size());

    // Sort scores in descending order
    if (!probs.m_is_sorted) {
        std::partial_sort(probs.m_probs.begin(), probs.m_probs.begin() + k, probs.m_probs.end(), std::greater<ProbIndex>{});
        probs.m_is_sorted = true;
    }
    if (k != probs.m_probs.size()) {
        probs.m_is_normalized = false;
    }
    probs.m_probs.resize(k);
}

void TopPSampler::apply(ProbArray &probs) {
    if (m_topp >= 1.0f) {
        return;
    }
    if (!probs.m_is_normalized || !probs.m_is_sorted) {
        softmax(probs);
    }

    // Compute the cumulative probabilities
    float cum_sum   = 0.0f;
    size_t last_idx = probs.m_probs.size();

    for (size_t i = 0; i < probs.m_probs.size(); ++i) {
        cum_sum += probs.m_probs[i].prob;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= m_topp && i + 1 >= m_min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    if (last_idx != probs.m_probs.size()) {
        probs.m_is_normalized = false;
    }
    probs.m_probs.resize(last_idx);
}

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

void StochasticSampler::apply(ProbArray &probs) {
    if (!probs.m_is_normalized || !probs.m_is_sorted) {
        softmax(probs);
    }

    std::discrete_distribution<int> dist(
        probs_iterator{probs.m_probs.data()}, probs_iterator{probs.m_probs.data() + probs.m_probs.size()}
    );
    std::mt19937 gen(m_seed);
    auto idx         = dist(gen);
    probs.m_probs[0] = probs.m_probs[idx];
    probs.m_probs.resize(1);
    probs.m_is_normalized = false;
}

void TemperatureExtSampler::apply(ProbArray &probs) {
    if (m_delta > 0) {
        const float min_temp = std::max(0.0f, m_temperature - m_delta);
        const float max_temp = m_temperature + m_delta;
        float exponent_val   = m_exponent;

        // no need to do anything if there is only one (or zero) candidates
        if (probs.m_probs.size() <= 1) {
            return;
        }

        // Calculate maximum possible entropy
        float max_entropy = -std::log(1.0f / probs.m_probs.size());

        if (!probs.m_is_normalized || !probs.m_is_sorted) {
            softmax(probs);
        }

        // Calculate entropy of the softmax probabilities
        float entropy = 0.0f;
        for (size_t i = 0; i < probs.m_probs.size(); ++i) {
            float prob = probs.m_probs[i].prob;
            if (prob > 0.0f) { // Ensure no log(0)
                entropy -= prob * std::log(prob);
            }
        }

        // Normalize the entropy (max_entropy cannot be 0 here because we checked cur_p->size != 1 above)
        float normalized_entropy = entropy / max_entropy;

        // Map the normalized entropy to the desired temperature range using the power function
        float dyn_temp = min_temp + (max_temp - min_temp) * std::pow(normalized_entropy, exponent_val);

        {
            // fmt::println("Your text maxtemp value is: {}", max_temp);
            // fmt::println("Entropy: {}", entropy);
            // fmt::println("Max Possible Entropy: {}", max_entropy);
            // fmt::println("Normalized Entropy: {}", normalized_entropy);
            // fmt::println("Exponent: {}", exponent_val);
            // fmt::println("Dynamic Temperature (dyn_temp): {}", dyn_temp);
        }

        // Re-compute softmax probabilities after scaling logits with dynamic temperature
        const double max_l_double = probs.m_probs[0].prob / dyn_temp;

        double cum_sum_double = 0.0;
        for (size_t i = 0; i < probs.m_probs.size(); ++i) {
            probs.m_probs[i].prob /= dyn_temp; // Apply the dynamically calculated temperature scaling
            double p              = std::exp(probs.m_probs[i].prob - max_l_double);
            probs.m_probs[i].prob = p; // Store the scaled probability
            cum_sum_double += p;
        }

        for (size_t i = 0; i < probs.m_probs.size(); ++i) {
            probs.m_probs[i].prob /= cum_sum_double; // Re-normalize the probabilities
        }
        probs.m_is_normalized = true;
        {
            // Print the updated top 25 probabilities after temperature scaling
            // fmt::println("\nUpdated Top 25 Probabilities After Dynamic Temperature Scaling (in percentages):");
            // for (size_t i = 0; i < 25 && i < probs.size(); ++i) {
            //     fmt::println("Token {}: {}%", i + 1, probs[i].prob * 100.0f);
            // }
        }

    } else {
        for (size_t i = 0; i < probs.m_probs.size(); ++i) {
            probs.m_probs[i].prob /= m_temperature;
        }
    }
}

void PenaltyChecker::apply(ProbArray &probs) {
    SMART_UNUSED(probs);
    if (m_ignore_eos) {
        // if ignore eos, set the logit of eos token to -INFINITY, so it will not be selected
        if (probs.m_probs.size() > (size_t)m_special_eos_id &&
            probs.m_probs[m_special_eos_id].index == m_special_eos_id) {
            probs.m_probs[m_special_eos_id].prob = -INFINITY;
        } else {
            // search and set the logit of eos token to -INFINITY
            for (size_t i = 0; i < probs.m_probs.size(); ++i) {
                if (probs.m_probs[i].index == m_special_eos_id) {
                    probs.m_probs[i].prob = -INFINITY;
                    break;
                }
            }
        }
    }

    if ((m_penalty_last_n == 0) || (m_penalty_repeat == 1.0f && m_penalty_freq == 0.0f && m_penalty_present == 0.0f)) {
        return;
    }

    bool nl_found  = false;
    size_t nl_idx  = 0;
    float nl_logit = -INFINITY;
    if (!m_penalize_nl) {
        // if not penalize nl, save its original logit value, so we can restore it later
        SMART_ASSERT(m_linefeed_id >= 0);

        // optimistically check if the candidates are not yet sorted/shuffled/truncated
        if (probs.m_probs.size() > (size_t)m_linefeed_id && probs.m_probs[m_linefeed_id].index == m_linefeed_id) {
            nl_found = true;
            nl_idx   = m_linefeed_id;
            nl_logit = probs.m_probs[m_linefeed_id].prob;
        } else {
            // else, search for the linefeed token
            for (size_t i = 0; i < probs.m_probs.size(); ++i) {
                if (probs.m_probs[i].index == m_linefeed_id) {
                    nl_found = true;
                    nl_idx   = i;
                    nl_logit = probs.m_probs[i].prob;
                    break;
                }
            }
        }
    }

    // Create a frequency map to count occurrences of each token in last_tokens
    // TODO: optimize this by maintaining the token count in the sampler context
    using llama_token_cnt = std::unordered_map<llama_token, int>;
    llama_token_cnt token_count;

    for (int i = 0; i < std::min<int>(m_penalty_last_n, m_prev.size()); ++i) {
        token_count[m_prev[i]]++;
    }

    // Apply frequency and presence penalties to the cur_p
    for (size_t i = 0; i < probs.m_probs.size(); ++i) {
        const auto token_iter = token_count.find(probs.m_probs[i].index);
        if (token_iter == token_count.end()) {
            continue;
        }

        const int count = token_iter->second;

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (probs.m_probs[i].prob <= 0) {
            probs.m_probs[i].prob *= m_penalty_repeat;
        } else {
            probs.m_probs[i].prob /= m_penalty_repeat;
        }

        probs.m_probs[i].prob -= float(count) * m_penalty_freq + float(count > 0) * m_penalty_present;
    }

    probs.m_is_sorted = false;

    if (!m_penalize_nl && nl_found) {
        // restore the logit of the newline token if it was penalized
        probs.m_probs[nl_idx].prob = nl_logit;
    }
}

void PenaltyChecker::accept(Tokenizer::Token token) {
    if (m_penalty_last_n > 0) {
        m_prev.push_back(token);
    }
}

} // namespace smart
