#include "sampler_chain.hpp"

namespace smart {

SamplerChain::SamplerChain(SamplerConfig config) : m_config(config) {
    m_penalties_checker = std::make_shared<PenalitiesChecker>(
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
    m_samplers.emplace_back(std::make_shared<TopkSampler>(config.topk));
    m_samplers.emplace_back(std::make_shared<ToppSampler>(config.topp));
    m_samplers.emplace_back(std::make_shared<TemperatureExtMapper>(config.temp));
    m_samplers.emplace_back(std::make_shared<SoftmaxMapper>());
    m_samplers.emplace_back(std::make_shared<GreedySampler>());
    // m_samplers.emplace_back(std::make_shared<DistSampler>(config.seed));
}

void SamplerChain::convert_logits(const std::vector<float> &logits) {
    m_probs.resize(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        m_probs[i].index = i;
        m_probs[i].prob  = logits[i];
    }
}

int SamplerChain::sample(std::vector<float> &logits) {
    convert_logits(logits);
    // for (size_t i = 0; i < std::min((size_t)10, m_probs.size()); i++) {
    //     fmt::println("[{}] <{}: {}>", i, m_probs[i].index, m_probs[i].prob);
    // }
    // fmt::println("--------------------------------");

    for (auto &sampler : m_samplers) {
        sampler->apply(m_probs);
        // for (size_t i = 0; i < std::min((size_t)10, m_probs.size()); i++) {
        //     fmt::println("[{}] <{}: {}>", i, m_probs[i].index, m_probs[i].prob);
        // }
        // fmt::println("--------------------------------");
    }
    m_penalties_checker->accept(Tokenizer::Token(m_probs[0].index));
    return m_probs[0].index;
}

void SamplerChain::apply(std::vector<ProbIndex> &probs) {
    SMART_UNUSED(probs);
    // for (auto &sampler : m_samplers) {
    //     sampler.apply(probs);
    // }
}

} // namespace smart
