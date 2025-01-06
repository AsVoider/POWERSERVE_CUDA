add_library(smart_serving STATIC)
target_include_directories(smart_serving PUBLIC .)

target_link_libraries(smart_serving PUBLIC
    fmt
    ggml
    llama_tokenizer
    nlohmann_json::nlohmann_json
    uv_a
)

if (SMART_WITH_PERFETTO)
    target_link_libraries(smart_serving PRIVATE perfetto)
endif()

if (NOT MSVC)
    target_compile_options(smart_serving PRIVATE
        -Wall
        -Wextra

        # -Wno-unused-function
    )

    if (SMART_ENABLE_WERROR)
        target_compile_options(smart_serving PRIVATE -Werror)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(smart_serving PRIVATE -Wno-unknown-pragmas)
endif()

add_subdirectory(storage)
add_subdirectory(backend)
add_subdirectory(core)
add_subdirectory(executor)
add_subdirectory(graph)
add_subdirectory(model)
add_subdirectory(sampler)
if (SMART_WITH_QNN)
    add_subdirectory(speculative)
endif()
add_subdirectory(tokenizer)
