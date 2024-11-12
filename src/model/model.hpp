#pragma once

#include "backend/platform.hpp"
#include "graph/graph.hpp"
#include "model/module/attention.hpp"
#include "model/module/ffn.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <string>

namespace smart {

struct Model {
public:
    std::string m_filename;
    std::shared_ptr<Config> m_config;
    std::shared_ptr<Weight> m_weights;
    std::shared_ptr<Attention> m_attn;
    std::shared_ptr<FFN> m_ffn;
    std::shared_ptr<Platform> m_platform;

public:
    Model(const std::string &filename) :
        m_filename(filename),
        m_config(nullptr),
        m_weights(nullptr),
        m_attn(nullptr),
        m_ffn(nullptr) {}

    virtual ~Model() = default;

public:
    virtual Graph *prefill() = 0;
    virtual Graph *decode()  = 0;

    virtual void generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps) = 0;
};

} // namespace smart
