#include "CLI/CLI.hpp"
#include "common/logger.hpp"
#include "simple_server.hpp"

#include <cstdlib>
#include <string>

// TODO:
// 1. signal handler & exception handle
// 2. completely support OpenAI API

int main(int argc, char *argv[]) {
    smart::print_timestamp();

    // 0. load config
    std::string model_folder;
    std::string host = "127.0.0.1";
    int port         = 8080;

    CLI::App app("Server program");

    app.add_option("--model-folder", model_folder);
    app.add_option("--host", host);
    app.add_option("--port", port);

    CLI11_PARSE(app, argc, argv);

    simple_server_handler(model_folder, host, port);

    return 0;
}
