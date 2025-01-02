#include "sampler_chain.hpp"

namespace smart {

SamplerChain::SamplerChain(const HyperParams::SamplerConfig &config, const Tokenizer &tokenizer) : m_config(config) {
    if (m_config.seed == 0) {
        std::random_device rd;
        m_config.seed = rd();
    }
    SMART_LOG_INFO("seed: {}", m_config.seed);

    // Samplers in order:
    // - Repeat penalty
    // - Top K
    // - Temperature
    // - Top P
    // - Stochastic

    m_samplers.emplace_back(std::make_unique<RepeatPenaltySampler>(
        tokenizer.n_vocabs(),
        tokenizer.m_vocab.special_eos_id,
        tokenizer.m_vocab.linefeed_id,
        config.penalty_last_n,
        config.penalty_repeat,
        config.penalty_freq,
        config.penalty_present,
        config.penalize_nl,
        config.ignore_eos
    ));
    m_samplers.emplace_back(std::make_unique<TopKSampler>(config.top_k));
    m_samplers.emplace_back(std::make_unique<TemperatureSampler>(config.temperature));
    m_samplers.emplace_back(std::make_unique<SoftmaxSampler>());
    m_samplers.emplace_back(std::make_unique<TopPSampler>(config.top_p));
    m_samplers.emplace_back(std::make_unique<NormalizeSampler>());
    m_samplers.emplace_back(std::make_unique<StochasticSampler>(config.seed));
}

void SamplerChain::apply(ProbArray &probs) {
    for (auto &sampler : m_samplers) {
        sampler->apply(probs);
    }
}

void SamplerChain::accept(Token token) {
    for (auto &sampler : m_samplers) {
        sampler->accept(token);
    }
}

} // namespace smart
