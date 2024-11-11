add_library(smart_serving STATIC)
target_include_directories(smart_serving PUBLIC .)

target_link_libraries(smart_serving PUBLIC
    fmt
    ggml
    llama_tokenizer
    uv_a
    nlohmann_json::nlohmann_json
)

target_sources(smart_serving PRIVATE
    "common.cpp"
)

add_subdirectory(backend)
add_subdirectory(core)
add_subdirectory(executor)
add_subdirectory(graph)
add_subdirectory(model)
add_subdirectory(sampler)
add_subdirectory(speculative)
add_subdirectory(tokenizer)

if (NOT MSVC)
    target_compile_options(smart_serving PRIVATE
        -Wall
        -Wextra
        -Werror

        -Wno-unused-function
    )
endif()
