#include "tokenizer.hpp"

#include "ggml.h"

namespace smart {

Tokenizer::Tokenizer(const Path &vocab_path) {
    struct ggml_context *ctx       = nullptr;
    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx      = &ctx,
    };

    struct gguf_context *meta = gguf_init_from_file(vocab_path.c_str(), params);
    SMART_ASSERT(meta);

    llm_load_vocab(m_vocab, meta);

    gguf_free(meta);

    debug_tokenizer();
}

size_t Tokenizer::n_vocabs() const {
    return m_vocab.n_vocab;
}

auto Tokenizer::bos_token() const -> Token {
    return m_vocab.special_bos_id;
}

auto Tokenizer::tokenize(const std::string &text, bool add_special) const -> std::vector<Token> {
    return llama_tokenize_internal(m_vocab, text, add_special, true);
}

auto Tokenizer::to_string(Token token, bool special) const -> std::string {
    return llama_token_to_piece(m_vocab, token, special);
}

} // namespace smart
