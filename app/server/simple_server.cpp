#include "simple_server.hpp"

#include "httplib.h"
#include "openai_api.hpp"

#include <memory>
#include <string>

SimpleServer::SimpleServer(
    const std::string &model_folder, const std::string &qnn_lib_folder, const std::string &host, const int port
) :
    m_server_context(model_folder, qnn_lib_folder) {
    // set up server
    m_server = std::make_unique<httplib::Server>();
    m_server->bind_to_port(host, port);

    const auto completion_handler = [this](const httplib::Request &request, httplib::Response &response) {
        handler_completion(m_server_context, request, response);
    };
    const auto chat_handler = [this](const httplib::Request &request, httplib::Response &response) {
        handler_chat(m_server_context, request, response);
    };
    const auto model_handler = [this](const httplib::Request &request, httplib::Response &response) {
        handler_model(m_server_context, request, response);
    };

    m_server->Post("/completion", completion_handler);
    m_server->Post("/completions", completion_handler);
    m_server->Post("/v1/completions", completion_handler);

    m_server->Post("/chat/completions", chat_handler);
    m_server->Post("/v1/chat/completions", chat_handler);

    m_server->Get("/v1/models", model_handler);

    SMART_LOG_INFO("server is listening http://{}:{}", host, port);
}

void simple_server_handler(
    const std::string &model_folder, const std::string &qnn_lib_folder, const std::string &host, const int port
) {
    SimpleServer server(model_folder, qnn_lib_folder, host, port);
    server.execute();
    server.join();
}
