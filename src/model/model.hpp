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
        std::deque<Token> m_tokens = {};
        size_t m_current_pos       = 0;

        Model &m_model;
        Tokenizer &m_tokenizer;
        Sampler &m_sampler;

    public:
        TokenIterator(
            Model &model,
            Tokenizer &tokenizer,
            Sampler &sampler,
            const std::string &prompt,
            int steps,
            size_t cur_pos = 0
        ) :
            n_reset(steps),
            m_prompt(prompt),
            m_tokens(),
            m_current_pos(cur_pos),
            m_model(model),
            m_tokenizer(tokenizer),
            m_sampler(sampler) {
            if (cur_pos >= (size_t)steps) {
                return;
            }

            auto prompt_tokens   = m_tokenizer.tokenize(m_prompt, m_tokenizer.m_vocab.tokenizer_add_bos);
            auto n_prompt_tokens = prompt_tokens.size();
            std::vector<Token> tokens;
            std::copy(prompt_tokens.begin(), std::prev(prompt_tokens.end()), std::back_inserter(tokens));
            int position = 0;
#if defined(SMART_WITH_QNN)
            auto &m_platform = m_model.m_platform;
            if (m_platform->qnn_backend) {
                m_platform->qnn_backend->m_causal_lm->kv_cache->truncate(
                    m_platform->qnn_backend->m_causal_lm->largest_chunks()[0]->m_config.kv_size
                );
                position = m_platform->qnn_backend->m_causal_lm->kv_cache->position;
            }
#endif
            std::vector<int> pos(n_prompt_tokens - 1); // FIXME: cpu need split small batch-size, else be oom
            std::iota(pos.begin(), pos.end(), position);
            // prefill
            m_model.decode(m_sampler, tokens, pos, false); // UNUSED ret
            m_current_pos = n_prompt_tokens - 1;           // TODO: get pos from kv interface
            m_tokens.push_back(prompt_tokens.back());
            // m_current_pos = m_model.m_platform->ggml_backend->m_kv->kv_cache->position;
#if defined(SMART_WITH_QNN)
            if (m_platform->qnn_backend) {
                m_current_pos = m_platform->qnn_backend->m_causal_lm->kv_cache->position;
            }
#endif
        }

        ~TokenIterator() = default;

        auto operator*() const -> Token {
            return m_tokens.front();
        }

        TokenIterator &operator++() {
            if (n_reset > 0 && m_tokens.size() >= 1) {
                std::vector<int> pos(1, m_current_pos);
                std::vector<int> token(1, m_tokens.front());
                auto ret = m_model.decode(m_sampler, token, pos, true);
                std::copy(ret.begin(), ret.end(), std::back_inserter(m_tokens));
                --n_reset;
                m_tokens.pop_front();
                ++m_current_pos;
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

    class TokenRange {
    public:
        size_t m_steps;
        std::string m_prompt;

        Model &m_model;
        Tokenizer &m_tokenizer;
        Sampler &m_sampler;

    public:
        TokenRange(Model &model, Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps) :
            m_steps(steps),
            m_prompt(prompt),
            m_model(model),
            m_tokenizer(tokenizer),
            m_sampler(sampler) {}

        ~TokenRange() = default;

    public:
        TokenIterator begin() const {
            return TokenIterator(m_model, m_tokenizer, m_sampler, m_prompt, m_steps, 0);
        }

        TokenIterator end() const {
            return TokenIterator(m_model, m_tokenizer, m_sampler, m_prompt, 0, 0);
        }
    };

public:
    std::string m_filename;
    std::shared_ptr<LLMConfig> m_config;
    std::shared_ptr<Weight> m_weights;
    std::shared_ptr<Attention> m_attn;
    std::shared_ptr<FFN> m_ffn;
    std::shared_ptr<Platform> m_platform;

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
    virtual auto generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps)
        -> TokenRange = 0;
};

} // namespace smart
