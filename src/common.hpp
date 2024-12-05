#pragma once

#include "fmt/base.h"
#include "fmt/ranges.h"
#include "fmt/std.h"
#include "uv.h"

#include <cstdlib>
#include <filesystem>

#if !defined(ALWAYS_INLINE)
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

#define SMART_UNUSED(x) ((void)(x))

#define SMART_BUILTIN_EXPECT(expr, value) __builtin_expect((expr), (value))

// #define SMART_NO_ASSERT

#if defined(SMART_NO_ASSERT)
#define SMART_ASSERT(expr) SMART_UNUSED(expr)
#else
#define SMART_ASSERT(expr)                                                                                             \
    do {                                                                                                               \
        if (SMART_BUILTIN_EXPECT(!(expr), 0)) {                                                                        \
            fflush(stdout);                                                                                            \
            fflush(stderr);                                                                                            \
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

void get_memory_usage(const std::string &msg = "");

inline void read_binary_file(const Path &path, uint8_t *buffer, size_t size) {
    uv_fs_t req;
    uv_fs_open(nullptr, &req, path.c_str(), O_RDONLY, 0, nullptr);
    auto fd = (uv_file)req.result;
    SMART_ASSERT(fd >= 0);
    uv_fs_req_cleanup(&req);

    uv_fs_fstat(nullptr, &req, fd, nullptr);
    SMART_ASSERT(req.result == 0);
    SMART_ASSERT(req.statbuf.st_size == size);
    uv_fs_req_cleanup(&req);

    uv_buf_t buf = {
        .base = (char *)buffer,
        .len  = size,
    };
    uv_fs_read(nullptr, &req, fd, &buf, 1, 0, nullptr);
    SMART_ASSERT((size_t)req.result == size);
    uv_fs_req_cleanup(&req);

    uv_fs_close(nullptr, &req, fd, nullptr);
    uv_fs_req_cleanup(&req);
}

template <typename T>
inline auto read_binary_file(const Path &path, size_t n_elements) -> std::vector<T> {
    std::vector<T> data(n_elements);
    read_binary_file(path, (uint8_t *)data.data(), n_elements * sizeof(T));
    return data;
}

inline void print_timestamp() {
    fmt::println(stderr, "Compiled on: {} at {}", __DATE__, __TIME__);
}
} // namespace smart
