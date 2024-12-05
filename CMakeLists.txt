cmake_minimum_required(VERSION 3.16)
project("SmartServing" C CXX)

option(SMART_ENABLE_ASAN "Enable address santizer" OFF)
option(SMART_ENABLE_UBSAN "Enable undefined santizer" OFF)
option(SMART_ENABLE_LTO "Enable link-time optimization" OFF)

option(SMART_WITH_QNN "Enable QNN backend" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

# Remove -DNDEBUG from the default Release flags
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

# Remove -DNDEBUG from the default RelWithDebInfo flags
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")

set(BUILD_SHARED_LIBS OFF)

if (SMART_ENABLE_ASAN)
    if (ANDROID)
        add_compile_options(-fsanitize=hwaddress -fno-omit-frame-pointer)
        link_libraries(-fsanitize=hwaddress)
    else()
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif()
endif()

if (SMART_ENABLE_UBSAN) # use with --target, not use full compilition
    add_compile_options(-fsanitize=undefined)
    link_libraries(-fsanitize=undefined)
    if (ANDROID)
        add_compile_options(-static-libsan)
        link_libraries(-static-libsan)
    endif()
endif()

if (SMART_ENABLE_LTO)
    if (${CMAKE_C_COMPILER_ID} MATCHES "Clang")
        add_compile_options(-flto=thin)
    else()
        add_compile_options(-flto)
    endif()
endif()

add_subdirectory(libs)
add_subdirectory(src)
add_subdirectory(bin)
add_subdirectory(tests)
add_subdirectory(tools)
