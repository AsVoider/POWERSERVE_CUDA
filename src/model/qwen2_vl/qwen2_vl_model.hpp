#pragma once

#include "backend/platform.hpp"
#include "ggml.h"
#include "model/model.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace smart {

struct Qwen2_VL : Model {
public:
    // ggml need those context
    ggml_context *ggml_ctx;
    gguf_context *gguf_ctx;
    int img_tokens_length = 0;
    std::vector<std::vector<float>> img_embedding;

public:
    explicit Qwen2_VL(
        const std::string &filename, const std::shared_ptr<Config> &config, const std::shared_ptr<Platform> &platform
    );
    ~Qwen2_VL() override;

public:
    // void generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps) override;
    auto decode(Sampler &sampler, const std::vector<Tokenizer::Token> tokens, const std::vector<int> pos, bool lm_head)
        -> std::vector<Tokenizer::Token> override;
    auto generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps) -> TokenRange override;
    auto preprocess(const Path &img_path, const std::string &prompt) -> std::string;
    auto forward(
        const std::vector<int> &tokens,
        const std::vector<int> &pos,
        const CausalAttentionMask &mask,
        bool lm_head = true
    ) -> std::vector<std::vector<float>>;
};

} // namespace smart
