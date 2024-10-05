#include "ggml.h"
#include <string>
#include "CLI/CLI.hpp"


int main(int argc, char *argv[]) {
    std::string file_path = "";

    CLI::App app("Demo program for llama3");
    
    app.add_option("--file-path", file_path)->required();
    CLI11_PARSE(app, argc, argv);

    ggml_context *ggml_ctx = nullptr;
    gguf_context *gguf_ctx = nullptr;

    gguf_init_params params = {
        .no_alloc = false,
        .ctx = &ggml_ctx
    };
    gguf_ctx = gguf_init_from_file(file_path.c_str(), params);

}