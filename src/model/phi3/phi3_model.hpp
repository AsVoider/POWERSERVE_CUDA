#pragma once

#include "backend/platform.hpp"
#include "ggml.h"
#include "graph/graph.hpp"
#include "model/model.hpp"
#include "model/module/attention.hpp"
#include "model/module/ffn.hpp"
#include "model/phi3/phi3_config.hpp"
#include "model/phi3/phi3_weight.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace smart {

struct Phi3Model : Model {
private:
    // ggml need those context
    ggml_context *ggml_ctx;
    gguf_context *gguf_ctx;
    std::shared_ptr<Platform> plat;

public:
    explicit Phi3Model(const std::string &filename, int n_threads = 1);
    ~Phi3Model() override;

public:
    Graph *prefill() override;
    Graph *decode() override;
    void generate(Tokenizer *tk, Sampler *sampler, std::string prompt, int steps) override;
    std::vector<float> forward(int token, int pos);
};

} // namespace smart
