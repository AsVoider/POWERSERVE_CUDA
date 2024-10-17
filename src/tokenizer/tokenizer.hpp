#pragma once

#include "common.hpp"
#include "llama-vocab.h"

namespace smart {

struct Tokenizer {
public:
    using Token = llama_vocab::id;

public:
    struct llama_vocab m_vocab;

public:
    explicit Tokenizer(const Path &vocab_path);
    ~Tokenizer() = default;

public:
    size_t n_vocabs() const;
    auto bos_token() const -> Token;
    auto tokenize(const std::string &text, bool add_special) const -> std::vector<Token>;
    auto to_string(Token token) const -> std::string;
};

} // namespace smart
