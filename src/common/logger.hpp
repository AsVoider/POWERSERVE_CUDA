#pragma once

#include "fmt/base.h"
#include "fmt/ranges.h"
#include "fmt/std.h"
#include "perf.hpp"

#include <cstdio>
#include <string>
#include <string_view>

#if !defined(ALWAYS_INLINE)
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

#define SMART_UNUSED(x) ((void)(x))

#define SMART_BUILTIN_EXPECT(expr, value) __builtin_expect((expr), (value))

// #define SMART_NO_ASSERT

#define SMART_LOG_DEBUG(...) fmt::println(stdout, "[DEBUG] " __VA_ARGS__)

#define SMART_LOG_INFO(...) fmt::println(stdout, "[INFO ] " __VA_ARGS__)

#define SMART_LOG_WARN(...) fmt::println(stderr, "[WARN ] " __VA_ARGS__)

#define SMART_LOG_ERROR(...) fmt::println(stderr, "[ERROR] " __VA_ARGS__)

#define SMART_ABORT(...)                                                                                               \
    do {                                                                                                               \
        fflush(stdout);                                                                                                \
        fflush(stderr);                                                                                                \
        SMART_LOG_ERROR("{}:{}: {}: Abort", __FILE__, __LINE__, __func__);                                             \
        SMART_LOG_ERROR("" __VA_ARGS__);                                                                               \
        SMART_LOG_ERROR("System error: {}", ::smart::get_system_error());                                              \
        abort();                                                                                                       \
    } while (0)

#if defined(SMART_NO_ASSERT)
#define SMART_ASSERT(expr) SMART_UNUSED(expr)
#else
#define SMART_ASSERT(expr, ...)                                                                                        \
    do {                                                                                                               \
        if (!(expr)) [[unlikely]] {                                                                                    \
            fflush(stdout);                                                                                            \
            fflush(stderr);                                                                                            \
            SMART_LOG_ERROR("{}:{}: {}: Assertion failed: {}", __FILE__, __LINE__, __func__, #expr);                   \
            SMART_LOG_ERROR("System error: {}", ::smart::get_system_error());                                          \
            SMART_LOG_ERROR("" __VA_ARGS__);                                                                           \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)
#endif

namespace smart {
inline namespace common {

inline void print_timestamp() {
    SMART_LOG_INFO("Compiled on: {} at {}", __DATE__, __TIME__);
}

inline std::string get_system_error() {
#ifdef _WIN32
    // TODO: include windows headers
    const DWORD error = getLastError();
    LPSTR buf;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        err,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&buf,
        0,
        NULL
    );
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
#else
    return {strerror(errno)};
#endif
}

inline std::string abbreviation(std::string text, size_t limit) {
    auto len = text.length();
    if (len > limit) {
        return fmt::format("{}...[omit {} chars]", text.substr(0, limit), len - limit);
    }
    return text;
}

// trim whitespace from the beginning and end of a string
inline std::string trim(const std::string &str) {
    size_t start = 0;
    size_t end   = str.size();
    while (start < end && isspace(str[start])) {
        start += 1;
    }
    while (end > start && isspace(str[end - 1])) {
        end -= 1;
    }
    return str.substr(start, end - start);
}

} // namespace common
} // namespace smart
