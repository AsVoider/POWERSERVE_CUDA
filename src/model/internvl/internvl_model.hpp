#pragma once

#include "ggml.h"
#include "model/model.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace smart {

struct InternVL : Model {
public:
    // ggml need those context
    ggml_context *ggml_ctx;
    gguf_context *gguf_ctx;
    bool lazy_load;
    std::vector<std::pair<int, size_t>> img_infos;
    std::vector<std::vector<float>> pixel_values_list;
    static const int IMG_START = 151646;

public:
    explicit InternVL(const std::string &filename, const std::shared_ptr<ModelConfig> &config);
    ~InternVL() override;

public:
    // void generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps) override;
    auto decode(Sampler &sampler, const std::vector<Token> tokens, const std::vector<int> pos, bool lm_head)
        -> std::vector<Token> override;
    auto generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps, size_t batch_size)
        -> TokenGenerator override;
    auto preprocess(const std::vector<Path> &img_paths, const std::string &prompt) -> std::string;
    auto forward(
        const std::vector<int> &tokens,
        const std::vector<int> &pos,
        const CausalAttentionMask &mask,
        bool lm_head = true
    ) -> std::vector<std::vector<float>> override;
};

} // namespace smart
