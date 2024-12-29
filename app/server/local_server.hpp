#pragma once

#include "concurrentqueue.h"
#include "server_handler.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

struct LocalRequest {
    std::string body;
};

struct LocalResponse;

class LocalServer {
public:
    using TaskQueue = moodycamel::ConcurrentQueue<LocalResponse *>;

    static constexpr auto DEFAULT_TASK_INTERVAL = std::chrono::milliseconds{100};

public:
    ServerContext m_context;

    TaskQueue m_task_queue;

    std::thread m_server_thread;

    std::atomic_flag m_stop;

public:
    LocalServer(
        const std::filesystem::path &model_folder, std::chrono::milliseconds task_interval = DEFAULT_TASK_INTERVAL
    );

    ~LocalServer() noexcept;

public:
    LocalResponse *create_completion_reponse(const LocalRequest &request);

    LocalResponse *create_chat_response(const LocalRequest &request);

    LocalResponse *create_model_response(const LocalRequest &request);

    std::optional<std::string> get_response(LocalResponse *response_ptr);

    bool poll_response(LocalResponse *response_ptr) const;

    void wait_response(LocalResponse *response_ptr) const;

    void destroy_response(LocalResponse *response_ptr);
};
