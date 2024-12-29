#pragma once

#include "httplib.h"
#include "server_handler.hpp"

#include <stdexcept>
#include <string>

struct SimpleServer {
private:
    ServerContext m_server_context;

    std::unique_ptr<httplib::Server> m_server;

    std::unique_ptr<std::thread> m_server_thread;

public:
    SimpleServer(const std::string model_folder, const std::string &host, const int port);

    ~SimpleServer() noexcept = default;

public:
    void execute() {
        if (m_server_thread) {
            throw std::runtime_error("there has already been one server thread");
        }
        m_server->listen_after_bind();
        m_server->wait_until_ready();
    }

    void join() {
        if (!m_server_thread) {
            throw std::runtime_error("the server hasn't been executed");
        }
        m_server_thread->join();
    }
};

void simple_server_handler(const std::string &model_folder, const std::string &host, const int port);
