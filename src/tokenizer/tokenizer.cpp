#include "tokenizer.hpp"

#include "ggml.h"

namespace smart {

Tokenizer::Tokenizer(const Path &vocab_path) {
	struct ggml_context *ctx	   = nullptr;
	struct gguf_init_params params = {
		.no_alloc = true,
		.ctx	  = &ctx,
	};

	struct gguf_context *meta = gguf_init_from_file(vocab_path.c_str(), params);
	SMART_ASSERT(meta);

	llm_load_vocab(vocab, meta);

	gguf_free(meta);
}

size_t Tokenizer::n_vocabs() const {
	return vocab.n_vocab;
}

auto Tokenizer::bos_token() const -> Token {
	return vocab.special_bos_id;
}

auto Tokenizer::tokenize(const std::string &text, bool add_special) const -> std::vector<Token> {
	return llama_tokenize_internal(vocab, text, add_special, true);
}

auto Tokenizer::to_string(Token token) const -> std::string {
	return llama_token_to_piece(vocab, token);
}

} // namespace smart
