#pragma once

#include "ggml.h"
#include "model/model.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace smart {

struct Qwen2Model : Model {
public:
    // ggml need those context
    ggml_context *ggml_ctx;
    gguf_context *gguf_ctx;
    bool lazy_load;

public:
    explicit Qwen2Model(const std::string &filename, const std::shared_ptr<ModelConfig> &config);
    ~Qwen2Model() override;

public:
    auto decode(Sampler &sampler, const std::vector<Token> tokens, const std::vector<int> pos, bool lm_head)
        -> std::vector<Token> override;
    auto generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps, size_t batch_size)
        -> TokenGenerator override;

    auto forward(
        const std::vector<int> &tokens,
        const std::vector<int> &pos,
        const CausalAttentionMask &mask,
        bool lm_head = true
    ) -> std::vector<std::vector<float>> override;
};

} // namespace smart
