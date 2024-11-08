#pragma once

#include "fmt/base.h"
#include "fmt/ranges.h"
#include "fmt/std.h"

#include <cstdlib>
#include <filesystem>

#define SMART_UNUSED(x) ((void)(x))

#define SMART_BUILTIN_EXPECT(expr, value) __builtin_expect((expr), (value))

// #define SMART_NO_ASSERT

#if defined(SMART_NO_ASSERT)
#define SMART_ASSERT(expr) SMART_UNUSED(expr)
#else
#define SMART_ASSERT(expr)                                                                                             \
    do {                                                                                                               \
        if (SMART_BUILTIN_EXPECT(!(expr), 0)) {                                                                        \
            fmt::println(stderr, "{}:{}: {}: Assertion failed: {}", __FILE__, __LINE__, __func__, #expr);              \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)
#endif

namespace smart {

using Path = std::filesystem::path;

static long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

#if defined(__ANDROID__)
static void get_memory_usage(std::string msg = "") {
    FILE *file = fopen("/proc/self/statm", "r");
    SMART_ASSERT(file != nullptr);

    size_t pages, resident, shared, text, lib, data, dt;
    fscanf(file, "%zu %zu %zu %zu %zu %zu %zu", &pages, &resident, &shared, &text, &lib, &data, &dt);

    fclose(file);

    long page_size = sysconf(_SC_PAGESIZE);
    size_t rss     = resident * page_size;
    size_t vms     = pages * page_size;

    fmt::println(stderr, "[{}] RSS: {} MB, VMS: {} MB", msg, rss / 1024 / 1024, vms / 1024 / 1024);
}
#elif defined(__linux__)
static void get_memory_usage(std::string msg = "") {
    FILE *file = fopen("/proc/self/status", "r");
    SMART_ASSERT(file != nullptr);
    char line[128];

    size_t rss = 0, vms = 0;
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line, "VmRSS: %zu", &rss);
            rss *= 1024; // Convert from KB to bytes
        }
        if (strncmp(line, "VmSize:", 7) == 0) {
            sscanf(line, "VmSize: %zu", &vms);
            vms *= 1024; // Convert from KB to bytes
        }
    }

    fclose(file);
    fmt::println(stderr, "[{}] RSS: {} MB, VMS: {} MB", msg, rss / 1024 / 1024, vms / 1024 / 1024);
}
#elif defined(__APPLE__)
static void get_memory_usage(std::string msg = "") {
    SMART_UNUSED(msg);
}
#elif defined(_WIN32)
static void get_memory_usage(std::string msg = "") {
    SMART_UNUSED(msg);
}
#endif

} // namespace smart
