#pragma once

namespace smart {

struct ProbIndex {
    float prob;
    int index;
};

struct Sampler{
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
};

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* sampler);
int sample(Sampler* sampler, float* logits);

} // namespace smart