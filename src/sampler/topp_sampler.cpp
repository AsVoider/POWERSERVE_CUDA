#include "topp_sampler.hpp"

#include "common.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>

namespace smart {

template <typename T>
void softmax(std::vector<T> &x) {
    // find max value (for numerical stability)
    auto max_val{*std::max_element(x.begin(), x.end())};

    // exp and sum
    std::transform(x.begin(), x.end(), x.begin(), [&](auto y) { return expf(y - max_val); });
    auto sum{std::accumulate(x.begin(), x.end(), 0.0f)};

    // normalize
    std::transform(x.begin(), x.end(), x.begin(), [&](auto y) { return y / sum; });
}

void softmax_(std::vector<float> &x) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (size_t i = 1; i < x.size(); i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (size_t i = 0; i < x.size(); i++) {
        x[i] /= sum;
    }
}

unsigned int random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(uint64_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(const std::vector<float> &probabilities, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (size_t i = 0; i < probabilities.size(); i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return probabilities.size() - 1; // in case of rounding errors
}

int sample_topp(const std::vector<float> &probabilities, float topp, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    auto n = probabilities.size();
    std::vector<ProbIndex> probindex;
    probindex.resize(n);

    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (size_t i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob  = probabilities[i];
            n0++;
        }
    }

    auto compare = [](const ProbIndex &a, const ProbIndex &b) { return a.prob > b.prob; };
    std::sort(probindex.begin(), probindex.begin() + n0, compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx          = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r   = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

int ToppSampler::sample(std::vector<float> &logits) {
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin{random_f32(&m_rng_state)};

    if (m_topp <= 0 || m_topp >= 1) {
        // simply sample from the predicted probability distribution
        return sample_mult(logits, coin);
    } else {
        // top-p (nucleus) sampling, clamping the least likely tokens to zero
        return sample_topp(logits, m_topp, coin);
    }
}

void ToppSampler::trans(std::vector<float> &logits) {
    if (m_temperature > 0) {
        // temperature scaling
        auto func = [&](auto x) { return x / m_temperature; };
        std::transform(logits.begin(), logits.end(), logits.begin(), func);
        // softmax
        softmax<float>(logits);
    }
}

} // namespace smart
