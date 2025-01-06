// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "model/model.hpp"
#include "speculative/token_tree.hpp"
#include "tokenizer/tokenizer.hpp"

#include <string>

namespace smart {

struct TreeSpeculative {
    TreeSpeculative(const ModelPtr &target_model, const ModelPtr &draft_model);

    void generate(const Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps);
    void print_stat();

private:
    ModelPtr target_model;
    ModelPtr draft_model;
    TokenTree token_tree;

    // Speculative decoding generates multiple tokens in one iteration.
    // We buffer these tokens in this queue, and pop tokens one by one, to adapt for TokenGenerator API.
    std::deque<Token> token_queue;

    void generate_tokens(const Tokenizer &tokenizer, Sampler &sampler, Token last_token);
};

} // namespace smart
