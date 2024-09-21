#pragma once

#include <cstdlib>
#include <filesystem>

#include "fmt/std.h"
#include "fmt/ranges.h"

#define SMART_UNUSED(x) ((void)(x))

#define SMART_BUILTIN_EXPECT(expr, value)  __builtin_expect((expr), (value))

// #define SMART_NO_ASSERT

#if defined(SMART_NO_ASSERT)
#define SMART_ASSERT(expr) SMART_UNUSED(expr)
#else
#define SMART_ASSERT(expr) \
    do { \
        if (SMART_BUILTIN_EXPECT(!(expr), 0)) { \
            fmt::println(stderr, "{}:{}: {}: Assertion failed: {}", __FILE__, __LINE__, __func__, #expr); \
            abort(); \
        } \
    } while(0)
#endif

using Path = std::filesystem::path;
