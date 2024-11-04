#pragma once

#include "sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace smart {

std::vector<ProbIndex> convert_logits(const std::vector<float> &logits);

struct SamplerConfig {
    uint64_t seed   = 0;
    float temp      = 0.80f;
    float top_p     = 0.95f; // 1.0 = disabled
    size_t top_k    = 40;
    size_t min_keep = 0; // 0 = disabled, otherwise samplers should return at least min_keep tokens

    // penalty parameters
    int32_t vocab_size              = 32000;
    Tokenizer::Token special_eos_id = 0;
    Tokenizer::Token linefeed_id    = 0;
    int penalty_last_n              = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float penalty_repeat            = 1.00f; // 1.0 = disabled
    float penalty_freq              = 0.00f; // 0.0 = disabled
    float penalty_present           = 0.00f; // 0.0 = disabled
    bool penalize_nl                = false; // consider newlines as a repeatable token
    bool ignore_eos                 = false;
};

struct SamplerChain : Sampler {
public:
    SamplerConfig m_config{};
    std::vector<std::shared_ptr<Sampler>> m_samplers{};
    std::shared_ptr<PenaltyChecker> m_penalties_checker{};

public:
    SamplerChain(SamplerConfig config);

    virtual ~SamplerChain() override = default;

public:
    void apply(std::vector<ProbIndex> &probs) override;
    void accept(ProbIndex &prob);
};

} // namespace smart