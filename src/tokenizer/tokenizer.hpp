#pragma once

#include "common/logger.hpp"
#include "common/type_def.hpp"
#include "llama-vocab.h"

#include <cstddef>
#include <string>

namespace smart {

struct ChatEntry {
    std::string role;
    std::string content;
};

struct Tokenizer {
public:
    struct llama_vocab m_vocab;

    std::string m_template_type;

public:
    explicit Tokenizer(const Path &vocab_path);
    ~Tokenizer() = default;

public:
    size_t n_vocabs() const;
    auto bos_token() const -> Token;
    auto tokenize(const std::string &text, bool add_special) const -> std::vector<Token>;
    auto to_string(Token token, bool special = true) const -> std::string;

    auto apply_chat_template(const std::vector<ChatEntry> &chat_history, const bool add_ass) const -> std::string;

public:
    void debug_tokenizer() {
        int vocab_size = (int)n_vocabs();

        auto print_tok([&](std::string name, llama_vocab::id token) {
            if (token < vocab_size && token >= 0)
                SMART_LOG_DEBUG("{:20}: {:6}: {}", name, token, to_string(token));
        });

        print_tok("special_bos", m_vocab.special_bos_id);
        print_tok("special_eos", m_vocab.special_eos_id);
        print_tok("special_unk", m_vocab.special_unk_id);
        print_tok("special_sep", m_vocab.special_sep_id);
        print_tok("special_pad", m_vocab.special_pad_id);
        print_tok("special_cls", m_vocab.special_cls_id);
        print_tok("special_mask", m_vocab.special_mask_id);
        print_tok("special_prefix", m_vocab.special_prefix_id);
        print_tok("special_suffix", m_vocab.special_suffix_id);
        print_tok("special_middle", m_vocab.special_middle_id);
        print_tok("special_eot", m_vocab.special_eot_id);
        print_tok("special_eom", m_vocab.special_eom_id);
    }
};

} // namespace smart
