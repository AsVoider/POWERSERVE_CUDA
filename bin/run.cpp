
#include "CLI/CLI.hpp"
#include "graph/graph.hpp"
#include "model/llama-impl/llama_model.hpp"
#include "sampler/greedy_sampler.hpp"
#include "tokenizer/tokenizer.hpp"
#include <string>

int main(int argc, char *argv[]) {
	// 0. load config
	std::string file_path		= "../models/Meta-Llama-3.1-8B/llama3-8b_Q4_0.gguf";
	std::string tokenizer_path	= "../models/Meta-Llama-3.1-8B/llama3.1_8b_vocab.gguf";
	float temperature			= 1.0f;		  // 0.0 = greedy deterministic. 1.0 = original. don't set higher
	float topp					= 0.9f;		  // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
	int steps					= 16;		  // number of steps to run for
	std::string prompt			= "One day,"; // prompt string
	unsigned long long rng_seed = 2024927;

	CLI::App app("Demo program for llama3");

	app.add_option("--file-path", file_path)->required();
	app.add_option("--vocab-path", tokenizer_path)->required();
	app.add_option("--prompt", prompt)->required();
	app.add_option("--steps", steps)->required();
	CLI11_PARSE(app, argc, argv);

	// 1. load model
	smart::LlamaModel model(file_path);

	// 2. load tokenizer
	smart::Tokenizer tokenizer(tokenizer_path);

	// 3. load sampler
	smart::GreedySampler sampler;

	// 4. generate
	model.generate(&tokenizer, (smart::Sampler *)(&sampler), prompt, steps);
	// 4.1 build graph
	// 4.2 sched graph
	// 4.2.1 run operators accordding to graph
}