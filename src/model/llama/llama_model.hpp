#pragma once

#include "ggml.h"
#include "graph/graph.hpp"
#include "model/llama/llama_config.hpp"
#include "model/llama/llama_weight.hpp"
#include "model/model.hpp"
#include "model/module/attention.hpp"
#include "model/module/ffn.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace smart {

struct LlamaModel : Model {
private:
    // ggml need those context
    ggml_context *ggml_ctx;
    gguf_context *gguf_ctx;

    std::shared_ptr<LlamaConfig> m_config;
    std::shared_ptr<LlamaWeight> m_weights;
    std::shared_ptr<Attention> m_attn;
    std::shared_ptr<FFN> m_ffn;

public:
    explicit LlamaModel(const std::string &filename);
    ~LlamaModel() override;

public:
    Graph *prefill() override;
    Graph *decode() override;
    void generate(Tokenizer *tk, Sampler *sampler, std::string prompt, int steps);
    std::vector<float> forward(int token, int pos);
};

} // namespace smart
