#pragma once

#include "common.hpp"

#include "llama-vocab.h"

namespace smart {

class Tokenizer {
public:
    using Token = llama_vocab::id;

    struct llama_vocab vocab;

    Tokenizer(const Path &vocab_path);

    size_t n_vocabs() const;
    auto bos_token() const -> Token;
    auto tokenize(const std::string &text, bool add_special) const -> std::vector<Token>;
    auto to_string(Token token) const -> std::string;
};

}
