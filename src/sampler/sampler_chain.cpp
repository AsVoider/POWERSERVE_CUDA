#include "sampler_chain.hpp"

namespace smart {

SamplerChain::SamplerChain(HyperParams::SamplerConfig config, const Tokenizer &tokenizer) : m_config(config) {
    m_samplers.emplace_back(std::make_shared<PenaltyChecker>(
        tokenizer.n_vocabs(),
        tokenizer.m_vocab.special_eos_id,
        tokenizer.m_vocab.linefeed_id,
        config.penalty_last_n,
        config.penalty_repeat,
        config.penalty_freq,
        config.penalty_present,
        config.penalize_nl,
        config.ignore_eos
    )); // TODO: the first or the last?
    m_samplers.emplace_back(std::make_shared<TopKSampler>(config.top_k));
    m_samplers.emplace_back(std::make_shared<TemperatureExtSampler>(config.temperature));
    m_samplers.emplace_back(std::make_shared<SoftmaxSampler>());
    m_samplers.emplace_back(std::make_shared<TopPSampler>(config.top_p));
    m_samplers.emplace_back(std::make_shared<NormalizeSampler>());
    m_samplers.emplace_back(std::make_shared<GreedySampler>());
}

SamplerChain::SamplerChain() {}

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
