#pragma once

#include "uv.h"

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

namespace smart {

struct ThreadConfig {
    std::vector<size_t> cpu_ids; // CPU affinity. Leave empty to create thread without setting affinity
};

struct ThreadPool {
public:
    using TaskFn = std::function<void(size_t thread_id)>;

public:
    ThreadPool(const std::vector<ThreadConfig> &configs);
    ~ThreadPool();

    // Sync with all threads in the thread pool
    // Only threads in the pool can call this method
    void barrier();

    // Run a task using all threads simultaneously
    // This method is blocking
    void run(TaskFn task);

    // Run a task using all threads simultaneously in the background
    void async_run(TaskFn task);

    // Wait for the last task to finish
    void wait();

    // Number of threads in the pool
    size_t size() const {
        return m_configs.size();
    }

private:
    std::vector<ThreadConfig> m_configs;
    std::atomic<bool> m_exited = false;
    uv_barrier_t m_run_barrier;
    uv_barrier_t m_sync_barrier;
    std::vector<std::thread> m_threads;
    TaskFn m_current_task = nullptr;

    void thread_main(size_t thread_id);
};

} // namespace smart
