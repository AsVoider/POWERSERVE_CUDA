#pragma once

#include "core/config.hpp"
#include "sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <vector>

namespace smart {

struct SamplerChain : Sampler {
    HyperParams::SamplerConfig m_config;
    std::vector<std::unique_ptr<Sampler>> m_samplers;

    SamplerChain(const HyperParams::SamplerConfig &config, const Tokenizer &tokenizer);

    virtual ~SamplerChain() override = default;

    void apply(ProbArray &probs) override;
    void accept(Token token) override;
};

} // namespace smart
