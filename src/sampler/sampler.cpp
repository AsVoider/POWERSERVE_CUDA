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
        float exponent_val   = m_exponent;

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
            double p      = exp(probs[i].prob - max_l_double);
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

void PenalitiesChecker::apply(std::vector<ProbIndex> &probs) {
    SMART_UNUSED(probs);
    if (m_ignore_eos) {
        // if ignore eos, set the logit of eos token to -INFINITY, so it will not be selected
        if (probs.size() > (size_t)m_special_eos_id && probs[m_special_eos_id].index == m_special_eos_id) {
            probs[m_special_eos_id].prob = -INFINITY;
        } else {
            // search and set the logit of eos token to -INFINITY
            for (size_t i = 0; i < probs.size(); ++i) {
                if (probs[i].index == m_special_eos_id) {
                    probs[i].prob = -INFINITY;
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
        if (probs.size() > (size_t)m_linefeed_id && probs[m_linefeed_id].index == m_linefeed_id) {
            nl_found = true;
            nl_idx   = m_linefeed_id;
            nl_logit = probs[m_linefeed_id].prob;
        } else {
            // else, search for the linefeed token
            for (size_t i = 0; i < probs.size(); ++i) {
                if (probs[i].index == m_linefeed_id) {
                    nl_found = true;
                    nl_idx   = i;
                    nl_logit = probs[i].prob;
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
    for (size_t i = 0; i < probs.size(); ++i) {
        const auto token_iter = token_count.find(probs[i].index);
        if (token_iter == token_count.end()) {
            continue;
        }

        const int count = token_iter->second;

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (probs[i].prob <= 0) {
            probs[i].prob *= m_penalty_repeat;
        } else {
            probs[i].prob /= m_penalty_repeat;
        }

        probs[i].prob -= float(count) * m_penalty_freq + float(count > 0) * m_penalty_present;
    }

    if (!m_penalize_nl && nl_found) {
        // restore the logit of the newline token if it was penalized
        probs[nl_idx].prob = nl_logit;
    }
}

void PenalitiesChecker::accept(Tokenizer::Token token) {
    if (m_penalty_last_n > 0) {
        m_prev.push_back(token);
    }
}

} // namespace smart
