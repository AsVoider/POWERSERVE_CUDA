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
    std::shared_ptr<Platform> plat;

public:
    explicit LlamaModel(const std::string &filename, int n_threads = 1);
    ~LlamaModel() override;

public:
    Graph *prefill() override;
    Graph *decode() override;
    void generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps) override;

    auto forward(int token, int pos) -> std::vector<float>;
};

} // namespace smart
