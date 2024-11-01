#include "sampler_chain.hpp"

namespace smart {

SamplerChain::SamplerChain(SamplerConfig config) : m_config(config) {
    m_samplers.emplace_back(std::make_unique<TopkSampler>(config.topk));
    m_samplers.emplace_back(std::make_unique<ToppSampler>(config.topp));
    m_samplers.emplace_back(std::make_unique<TemperatureExtMapper>(config.temp));
    m_samplers.emplace_back(std::make_unique<SoftmaxMapper>());
    // m_samplers.emplace_back(std::make_unique<GreedySampler>());
    m_samplers.emplace_back(std::make_unique<DistSampler>(config.seed));
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
    return m_probs[0].index;
}

void SamplerChain::apply(std::vector<ProbIndex> &probs) {
    SMART_UNUSED(probs);
    // for (auto &sampler : m_samplers) {
    //     sampler.apply(probs);
    // }
}

} // namespace smart
