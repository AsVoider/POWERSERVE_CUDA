#include "CLI/CLI.hpp"
#include "common/logger.hpp"
#include "simple_server.hpp"

#ifdef SMART_WITH_QNN
#include "backend/qnn/config.hpp"
#endif // SMART_WITH_QNN

#include <cstdlib>
#include <string>

// TODO:
// 1. signal handler & exception handle
// 2. completely support OpenAI API

int main(int argc, char *argv[]) {
    smart::print_timestamp();

    // 0. load config
    std::string model_folder;
    std::string qnn_lib_folder;
    std::string host = "127.0.0.1";
    int port         = 8080;

    CLI::App app("Server program");

    app.add_option("--model-folder", model_folder);
    app.add_option("--lib-folder", qnn_lib_folder);
    app.add_option("--host", host);
    app.add_option("--port", port);

    CLI11_PARSE(app, argc, argv);

#ifdef SMART_WITH_QNN
    if (qnn_lib_folder.empty()) {
        qnn_lib_folder = std::filesystem::path(model_folder) / smart::qnn::QNN_LIB_DIR_NAME;
    }
#endif // SMART_WITH_QNN
    simple_server_handler(model_folder, qnn_lib_folder, host, port);

    return 0;
}
