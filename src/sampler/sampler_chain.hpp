#pragma once

#include "core/config.hpp"
#include "sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cmath>
#include <vector>

namespace smart {

struct SamplerChain : Sampler {
public:
    HyperParams::SamplerConfig m_config;
    std::vector<std::shared_ptr<Sampler>> m_samplers;

public:
    SamplerChain(HyperParams::SamplerConfig config, const Tokenizer &tokenizer);
    SamplerChain();

    virtual ~SamplerChain() override = default;

public:
    void apply(ProbArray &probs) override;
    void accept(Token token) override;
};

} // namespace smart
