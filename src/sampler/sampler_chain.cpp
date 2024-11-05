#include "sampler_chain.hpp"

namespace smart {

SamplerChain::SamplerChain(SamplerConfig config) : m_config(config) {
    m_penalties_checker = std::make_shared<PenaltyChecker>(
        config.vocab_size,
        config.special_eos_id,
        config.linefeed_id,
        config.penalty_last_n,
        config.penalty_repeat,
        config.penalty_freq,
        config.penalty_present,
        config.penalize_nl,
        config.ignore_eos
    );

    m_samplers.emplace_back(m_penalties_checker);
    m_samplers.emplace_back(std::make_shared<TopKSampler>(config.top_k));
    m_samplers.emplace_back(std::make_shared<TopPSampler>(config.top_p));
    m_samplers.emplace_back(std::make_shared<TemperatureExtSampler>(config.temp));
    m_samplers.emplace_back(std::make_shared<SoftmaxSampler>());
    // m_samplers.emplace_back(std::make_shared<GreedySampler>());
    m_samplers.emplace_back(std::make_shared<StochasticSampler>(config.seed));
}

SamplerChain::SamplerChain() {}

void SamplerChain::apply(ProbArray &probs) {
    for (size_t i = 0; i < std::min((size_t)5, probs.m_probs.size()); i++) {
        fmt::println(stderr, "[{}] {}: {}", i, probs.m_probs[i].index, probs.m_probs[i].prob);
    }
    fmt::println(stderr, "----------");
    for (auto &sampler : m_samplers) {
        sampler->apply(probs);
        for (size_t i = 0; i < std::min((size_t)5, probs.m_probs.size()); i++) {
            fmt::println(stderr, "[{}] {}: {}", i, probs.m_probs[i].index, probs.m_probs[i].prob);
        }
        fmt::println(stderr, "----------");
    }
}

void SamplerChain::accept(Tokenizer::Token token) {
    m_penalties_checker->accept(token);
}

} // namespace smart
