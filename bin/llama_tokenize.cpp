#include "CLI/CLI.hpp"
#include "llama_tokenizer.hpp"

int main(int argc, char *argv[]) {
	Path vocab_path;
	std::string text;

	CLI::App app("Test program for LlamaTokenizer");
	app.add_option("--vocab-path", vocab_path)->required();
	app.add_option("--text", text)->required();
	CLI11_PARSE(app, argc, argv);

	fmt::println("{}", vocab_path);
	smart::LlamaTokenizer tokenizer(vocab_path);

	fmt::println("#vocab: {}", tokenizer.n_vocabs());
	fmt::println("BOS token: {}", tokenizer.bos_token());

	auto tokens = tokenizer.tokenize(text, true);
	fmt::println("{}", tokens);

	for (auto token : tokens) {
		fmt::print("{}", tokenizer.to_string(token));
	}
	fmt::println("");

	return 0;
}
