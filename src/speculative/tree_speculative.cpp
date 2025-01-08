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

#include "speculative/tree_speculative.hpp"

#include "core/perfetto_trace.hpp"
#include "core/timer.hpp"

constexpr bool debug_token_tree = false;

namespace powerserve {

TreeSpeculative::TreeSpeculative(const ModelPtr &target_model, const ModelPtr &draft_model) :
    target_model(target_model),
    draft_model(draft_model) {}

void TreeSpeculative::generate(const Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps) {
    auto prompt_tokens           = tokenizer.tokenize(prompt, tokenizer.m_vocab.tokenizer_add_bos);
    const size_t n_prompt_tokens = prompt_tokens.size();
    POWERSERVE_ASSERT(n_prompt_tokens >= 1);

    POWERSERVE_ASSERT(target_model->kv_cache->position == draft_model->kv_cache->position);
    size_t position = target_model->kv_cache->position;

    fmt::print("{}", prompt);
    fflush(stdout);

    {
        // Last prompt token is excluded in prefill, and will be used as the initiating token for decoding phase.
        const size_t n_prefill_tokens = n_prompt_tokens - 1;

        std::vector<Token> prefill_tokens(prompt_tokens.begin(), prompt_tokens.begin() + n_prefill_tokens);

        std::vector<int> prefill_positions(n_prefill_tokens);
        std::iota(prefill_positions.begin(), prefill_positions.end(), position);

        CausalAttentionMask prefill_attention_mask(n_prefill_tokens);

        PerfettoTrace::begin("target_model_prefill");
        target_model->forward(prefill_tokens, prefill_positions, prefill_attention_mask, false);
        PerfettoTrace::end();

        PerfettoTrace::begin("draft_model_prefill");
        draft_model->forward(prefill_tokens, prefill_positions, prefill_attention_mask, false);
        PerfettoTrace::end();

        position += n_prefill_tokens;
    }

    size_t n_generated_tokens = 0;
    size_t generation_time_ns = 0;
    auto last_token           = prompt_tokens[n_prompt_tokens - 1];
    Timer timer;
    while (position < n_prompt_tokens + steps) {
        if (token_queue.empty()) {
            timer.reset();
            generate_tokens(tokenizer, sampler, last_token);
            generation_time_ns += timer.elapsed_time_ns();

            POWERSERVE_ASSERT(token_queue.size() > 0);
            n_generated_tokens += token_queue.size();
        }

        last_token = token_queue.front();
        token_queue.pop_front();
        auto piece = tokenizer.to_string(last_token);
        fmt::print("{}", piece);
        fflush(stdout);
        position++;

        if (tokenizer.should_stop(last_token)) {
            fmt::println("[end of text]");
            fflush(stdout);
            break;
        }
    }

    fmt::print("\n");
    fmt::println(
        "Generation speed: {} tokens, {:.3f} ms, {:.3f} tokens/s",
        n_generated_tokens,
        generation_time_ns / 1e6,
        n_generated_tokens * 1e9 / generation_time_ns
    );
}

void TreeSpeculative::generate_tokens(const Tokenizer &tokenizer, Sampler &sampler, Token last_token) {
    constexpr size_t draft_batch_size = 16;

    token_tree.draft(draft_model, tokenizer, draft_batch_size, last_token);

    CausalAttentionMask mask(draft_batch_size, token_tree.attention_mask());

    PerfettoTrace::begin("target_model_forward");
    auto ret = target_model->forward(token_tree.tokens(), token_tree.positions(), mask);
    PerfettoTrace::end();

    target_model->kv_cache->rollback_tokens(draft_batch_size);

    token_tree.verify(target_model, draft_model, sampler, ret.logits_vector, [this](Token token) {
        token_queue.push_back(token);
    });

    if constexpr (debug_token_tree) {
        fmt::print("\n");
        token_tree.print_tree(tokenizer);
    }
}

void TreeSpeculative::print_stat() {
    token_tree.print_stat();
}

} // namespace powerserve
