#pragma once

#include <vector>
namespace smart {


struct Sampler{

    struct ProbIndex {
        float prob;
        int index;
    };

    int vocab_size;
    std::vector<ProbIndex> probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;

    Sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed) 
        : vocab_size(vocab_size),
          temperature(temperature),
          topp(topp),
          rng_state(rng_seed)
    {
        probindex = std::vector<ProbIndex>(vocab_size);
    }

    ~Sampler() {}

    int sample(float* logits);
};

} // namespace smart