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
#include "model/model.hpp"
#include "speculative/token_tree.hpp"
#include "tokenizer/tokenizer.hpp"

namespace powerserve {

struct SpeculativeModel {
public:
    struct TokenIterator {
    public:
        size_t n_reset             = 0;
        std::string m_prompt       = "";
        size_t m_batch_size        = 1;
        std::deque<Token> m_tokens = {};

        const ModelPtr target_model;
        const ModelPtr draft_model;
        const Tokenizer &m_tokenizer;
        Sampler &m_sampler;
        SpeculativeConfig &config;

    public:
        TokenIterator(
            TokenTree &token_tree,
            const ModelPtr &target_model,
            const ModelPtr &draft_model,
            const Tokenizer &tokenizer,
            Sampler &sampler,
            SpeculativeConfig &config,
            const std::string &prompt,
            size_t steps,
            size_t batch_size
        ) :
            n_reset(steps),
            m_prompt(prompt),
            m_batch_size(batch_size),
            m_tokens(),
            target_model(target_model),
            draft_model(draft_model),
            m_tokenizer(tokenizer),
            m_sampler(sampler),
            config(config),
            token_tree(token_tree) {
            if (steps <= 0) {
                return;
            }
            POWERSERVE_UNUSED(batch_size);
            auto prompt_tokens           = tokenizer.tokenize(prompt, tokenizer.m_vocab.tokenizer_add_bos);
            const size_t n_prompt_tokens = prompt_tokens.size();
            POWERSERVE_ASSERT(n_prompt_tokens >= 1);

            POWERSERVE_ASSERT(target_model->kv_cache->position == draft_model->kv_cache->position);
            size_t position = target_model->kv_cache->position;

            const size_t n_prefill_tokens = n_prompt_tokens - 1;

            std::vector<Token> prefill_tokens(prompt_tokens.begin(), prompt_tokens.begin() + n_prefill_tokens);

            std::vector<int> prefill_positions(n_prefill_tokens);
            std::iota(prefill_positions.begin(), prefill_positions.end(), position);

            CausalAttentionMask prefill_attention_mask(n_prefill_tokens);
            target_model->forward(prefill_tokens, prefill_positions, prefill_attention_mask, false);
            draft_model->forward(prefill_tokens, prefill_positions, prefill_attention_mask, false);

            position = target_model->kv_cache->position;
            m_tokens.push_back(prompt_tokens.back());
        }

        ~TokenIterator() = default;

        auto operator*() const -> Token {
            return m_tokens.front();
        }

        TokenIterator &operator++() {
            // TODO: decode
            if (n_reset > 0 && m_tokens.size() >= 1) {
                if (m_tokens.size() == 1) {
                    size_t position = target_model->kv_cache->position;
                    auto last_token = m_tokens.back();
                    generate_tokens(m_tokenizer, m_sampler, last_token);
                    POWERSERVE_ASSERT(token_queue.size() > 0);
                    for (auto token : token_queue) {
                        m_tokens.push_back(token);
                    }
                    token_queue.clear();
                }

                --n_reset;
                m_tokens.pop_front();
            }

            return *this;
        }

        bool operator!=(const TokenIterator &other) const {
            // simple unequal
            return m_prompt != other.m_prompt || n_reset != other.n_reset;
        }

        bool operator==(const TokenIterator &other) const {
            return !(*this != other);
        }

    private:
        TokenTree &token_tree;
        std::deque<Token> token_queue;

        void generate_tokens(const Tokenizer &tokenizer, Sampler &sampler, Token last_token) {
            token_tree.draft(draft_model, tokenizer, config.draft_batch_size, last_token);

            CausalAttentionMask mask(config.draft_batch_size, token_tree.attention_mask());

            auto ret = target_model->forward(token_tree.tokens(), token_tree.positions(), mask);

            target_model->kv_cache->rollback_tokens(config.draft_batch_size);

            token_tree.verify(target_model, draft_model, sampler, ret.logits_vector, [this](Token token) {
                token_queue.push_back(token);
            });

            if (config.token_tree.debug) {
                fmt::print("\n");
                token_tree.print_tree(tokenizer);
            }
        }
    };

    class TokenGenerator {
    public:
        size_t m_steps;
        std::string m_prompt;
        size_t m_batch_size;

        TokenTree &token_tree;
        const ModelPtr target_model;
        const ModelPtr draft_model;
        const Tokenizer &m_tokenizer;
        Sampler &m_sampler;
        SpeculativeConfig &config;

    public:
        TokenGenerator(
            TokenTree &token_tree,
            const ModelPtr &target_model,
            const ModelPtr &draft_model,
            const Tokenizer &tokenizer,
            Sampler &sampler,
            SpeculativeConfig &config,
            const std::string &prompt,
            int steps,
            size_t batch_size
        ) :
            m_steps(steps),
            m_prompt(prompt),
            m_batch_size(batch_size),
            token_tree(token_tree),
            target_model(target_model),
            draft_model(draft_model),
            m_tokenizer(tokenizer),
            m_sampler(sampler),
            config(config) {}

        ~TokenGenerator() = default;

    public:
        TokenIterator begin() const {
            return TokenIterator(
                token_tree, target_model, draft_model, m_tokenizer, m_sampler, config, m_prompt, m_steps, m_batch_size
            );
        }

        TokenIterator end() const {
            return TokenIterator(
                token_tree, target_model, draft_model, m_tokenizer, m_sampler, config, m_prompt, 0, m_batch_size
            );
        }
    };

public:
    const ModelPtr target_model;
    const ModelPtr draft_model;
    SpeculativeConfig config;

    SpeculativeModel(const ModelPtr &target_model, const ModelPtr &draft_model, const SpeculativeConfig &config) :
        target_model(target_model),
        draft_model(draft_model),
        config(config),
        token_tree(config) {}

    ~SpeculativeModel() = default;

public:
    auto generate(const Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps, size_t batch_size)
        -> TokenGenerator {
        return TokenGenerator(
            token_tree, target_model, draft_model, tokenizer, sampler, config, prompt, steps, batch_size
        );
    }

    void print_stat() {
        token_tree.print_stat();
    }

private:
    TokenTree token_tree;
};

} // namespace powerserve
