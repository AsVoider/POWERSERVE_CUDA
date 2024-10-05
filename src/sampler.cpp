#include "sampler.hpp"

namespace smart {

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int Sampler::sample(float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    next = sample_argmax(logits, vocab_size);
    return next;
}

} // namespace smart