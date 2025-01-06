// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "fmt/base.h"
#include "fmt/ranges.h"
#include "fmt/std.h"
#include "perf.hpp"

#include <cstdio>
#include <string>
#include <string_view>

// #define SMART_NO_ASSERT

#define SMART_LOG_DEBUG(...) fmt::println(stdout, "[DEBUG] " __VA_ARGS__)

#define SMART_LOG_INFO(...) fmt::println(stdout, "[INFO ] " __VA_ARGS__)

#define SMART_LOG_WARN(...) fmt::println(stderr, "[WARN ] " __VA_ARGS__)

#define SMART_LOG_ERROR(...) fmt::println(stderr, "[ERROR] " __VA_ARGS__)

#define SMART_LOG_EMPTY_LINE()                                                                                         \
    {                                                                                                                  \
        fflush(stdout);                                                                                                \
        fflush(stderr);                                                                                                \
        fmt::println(stdout, "");                                                                                      \
    }

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

} // namespace smart
