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

#include "backend/platform.hpp"
#include "model/module/attention.hpp"
#include "model/module/ffn.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <string>

namespace smart {

struct Model {
public:
    struct TokenIterator {
    public:
        size_t n_reset             = 0;
        std::string m_prompt       = "";
        size_t m_batch_size        = 1;
        std::deque<Token> m_tokens = {};

        Model &m_model;
        const Tokenizer &m_tokenizer;
        Sampler &m_sampler;

    public:
        TokenIterator(
            Model &model,
            const Tokenizer &tokenizer,
            Sampler &sampler,
            const std::string &prompt,
            int steps,
            size_t batch_size
        ) :
            n_reset(steps),
            m_prompt(prompt),
            m_batch_size(batch_size),
            m_tokens(),
            m_model(model),
            m_tokenizer(tokenizer),
            m_sampler(sampler) {
            if (steps <= 0) {
                return;
            }

            auto prompt_tokens   = m_tokenizer.tokenize(m_prompt, m_tokenizer.m_vocab.tokenizer_add_bos);
            auto n_prompt_tokens = prompt_tokens.size();
            size_t n_prefilled   = 0;
            size_t position      = 0;

            auto &m_platform = m_model.m_platform;
            auto &model_id   = m_model.m_config->model_id;
            m_platform->reset_kv_position(model_id);
            position = m_platform->get_kv_position(model_id);

            // prefill
            while (n_prefilled < n_prompt_tokens - 1) {
                size_t bs = std::min(m_batch_size, n_prompt_tokens - n_prefilled - 1);
                std::vector<Token> tokens;
                std::copy(
                    prompt_tokens.begin() + n_prefilled,
                    prompt_tokens.begin() + n_prefilled + bs,
                    std::back_inserter(tokens)
                );
                std::vector<int> pos(bs);
                std::iota(pos.begin(), pos.end(), position);
                m_model.decode(m_sampler, tokens, pos, false);
                position = m_platform->get_kv_position(model_id);
                n_prefilled += bs;
            }
            position = m_platform->get_kv_position(model_id);
            m_tokens.push_back(prompt_tokens.back());
        }

        ~TokenIterator() = default;

        auto operator*() const -> Token {
            return m_tokens.front();
        }

        TokenIterator &operator++() {
            if (n_reset > 0 && m_tokens.size() >= 1) {
                auto &platform   = m_model.m_platform;
                auto current_pos = platform->get_kv_position(m_model.m_config->model_id);
                std::vector<int> pos(1, current_pos);
                std::vector<int> token(1, m_tokens.front());
                auto ret = m_model.decode(m_sampler, token, pos, true);
                std::copy(ret.begin(), ret.end(), std::back_inserter(m_tokens));
                --n_reset;
                m_tokens.pop_front();
            }
            return *this;
        }

        bool operator!=(const TokenIterator &other) const {
            // sample unequal
            return m_prompt != other.m_prompt || n_reset != other.n_reset;
        }

        bool operator==(const TokenIterator &other) const {
            return !(*this != other);
        }
    };

    class TokenGenerator {
    public:
        size_t m_steps;
        std::string m_prompt;
        size_t m_batch_size;

        Model &m_model;
        const Tokenizer &m_tokenizer;
        Sampler &m_sampler;

    public:
        TokenGenerator(
            Model &model,
            const Tokenizer &tokenizer,
            Sampler &sampler,
            const std::string &prompt,
            int steps,
            size_t batch_size
        ) :
            m_steps(steps),
            m_prompt(prompt),
            m_batch_size(batch_size),
            m_model(model),
            m_tokenizer(tokenizer),
            m_sampler(sampler) {}

        ~TokenGenerator() = default;

    public:
        TokenIterator begin() const {
            return TokenIterator(m_model, m_tokenizer, m_sampler, m_prompt, m_steps, m_batch_size);
        }

        TokenIterator end() const {
            return TokenIterator(m_model, m_tokenizer, m_sampler, m_prompt, 0, m_batch_size);
        }
    };

public:
    std::string m_filename;
    std::shared_ptr<ModelConfig> m_config;
    std::shared_ptr<Weight> m_weights;
    std::shared_ptr<Attention> m_attn;
    std::shared_ptr<FFN> m_ffn;
    std::shared_ptr<Platform> m_platform;
    KVCacheInterface *kv_cache = nullptr;

public:
    Model(const std::string &filename) :
        m_filename(filename),
        m_config(nullptr),
        m_weights(nullptr),
        m_attn(nullptr),
        m_ffn(nullptr),
        m_platform(nullptr) {}

    virtual ~Model() = default;

    virtual auto forward(
        const std::vector<int> &tokens,
        const std::vector<int> &pos,
        const CausalAttentionMask &mask,
        bool lm_head = true
    ) -> std::vector<std::vector<float>> = 0;

public:
    virtual auto decode(Sampler &sampler, const std::vector<Token> tokens, const std::vector<int> pos, bool lm_head)
        -> std::vector<Token> = 0;
    virtual auto generate(
        const Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps, size_t batch_size
    ) -> TokenGenerator = 0;
};

using ModelPtr = std::shared_ptr<Model>;

} // namespace smart
