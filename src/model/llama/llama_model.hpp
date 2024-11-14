#pragma once

#include "backend/platform.hpp"
#include "ggml.h"
#include "graph/graph.hpp"
#include "model/model.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace smart {

struct LlamaModel : Model {
public:
    // ggml need those context
    ggml_context *ggml_ctx;
    gguf_context *gguf_ctx;

public:
    explicit LlamaModel(
        const std::string &filename, const std::shared_ptr<Config> &config, const std::shared_ptr<Platform> &platform
    );
    ~LlamaModel() override;

public:
    Graph *prefill() override;
    Graph *decode() override;
    void generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps) override;

    auto forward(
        const std::vector<int> &tokens,
        const std::vector<int> &pos,
        const CausalAttentionMask &mask,
        bool lm_head = true
    ) -> std::vector<std::vector<float>>;
};

} // namespace smart
