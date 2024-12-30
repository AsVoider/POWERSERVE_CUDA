#pragma once

#include "fmt/core.h"
#include "fmt/format.h"

#include <chrono>
#include <cstddef>

namespace smart {

inline namespace common {

struct CPUPerfResult {
    size_t total_user_time;
    size_t total_user_low_time;
    size_t total_system_time;
    size_t total_idle_time;
    size_t total_irq_time;
    size_t total_softirq_time;

    CPUPerfResult &operator-=(const CPUPerfResult &other) {
        total_user_time -= other.total_user_time;
        total_user_low_time -= other.total_user_low_time;
        total_system_time -= other.total_system_time;
        total_idle_time -= other.total_idle_time;
        total_irq_time -= other.total_irq_time;
        total_softirq_time -= other.total_softirq_time;
        return *this;
    }

    CPUPerfResult operator-(const CPUPerfResult &other) const {
        CPUPerfResult ret = *this;
        ret -= other;
        return ret;
    }

    std::string to_string() const {
        return fmt::format(
            "[CPU times(us)] user: {}; user_low: {}, system: {}, idle: {}, irq: {}, softirq: {}",
            total_user_time,
            total_user_low_time,
            total_system_time,
            total_idle_time,
            total_irq_time,
            total_softirq_time
        );
    }
};

struct IOPerfResult {
    // In bytes
    size_t total_bytes_read;
    // In bytes
    size_t total_bytes_write;

    IOPerfResult &operator-=(const IOPerfResult &other) {
        total_bytes_read -= other.total_bytes_read;
        total_bytes_write -= other.total_bytes_write;
        return *this;
    }

    IOPerfResult operator-(const IOPerfResult &other) const {
        IOPerfResult ret = *this;
        ret -= other;
        return ret;
    }

    std::string to_string() const {
        return fmt::format(
            "[I/O throughput(MB)] total read: {}, total write: {}",
            total_bytes_read / 1024 / 1024,
            total_bytes_write / 1024 / 1024
        );
    }
};

struct MemPerfResult {
    // In bytes
    size_t virtual_memory_size = 0;
    // In bytes
    size_t resident_set_size = 0;

    std::string to_string() const {
        return fmt::format(
            "[Memory(MB)] VMS: {}, RSS: {}", virtual_memory_size / 1024 / 1024, resident_set_size / 1024 / 1024
        );
    }
};

CPUPerfResult perf_get_cpu_result();

IOPerfResult perf_get_io_result();

MemPerfResult perf_get_mem_result();

struct TimeCounter {
public:
    std::chrono::steady_clock::time_point start;

public:
    TimeCounter() {
        start = std::chrono::steady_clock::now();
    }

    ~TimeCounter() noexcept = default;

public:
    size_t get_time_in_ms() const {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    void reset() {
        start = std::chrono::steady_clock::now();
    }
};

static long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

} // namespace common

} // namespace smart

template <>
struct fmt::formatter<smart::CPUPerfResult> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(const smart::CPUPerfResult &data, FormatContext &ctx) const {
        return formatter<std::string>::format(data.to_string(), ctx);
    }
};

template <>
struct fmt::formatter<smart::IOPerfResult> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(const smart::IOPerfResult &data, FormatContext &ctx) const {
        return formatter<std::string>::format(data.to_string(), ctx);
    }
};

template <>
struct fmt::formatter<smart::MemPerfResult> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(const smart::MemPerfResult &data, FormatContext &ctx) const {
        return formatter<std::string>::format(data.to_string(), ctx);
    }
};
