add_library(smart_serving STATIC)
target_link_libraries(smart_serving PUBLIC fmt::fmt ggml llama_tokenizer)
target_include_directories(smart_serving PUBLIC .)

add_subdirectory(backend)
add_subdirectory(core)
add_subdirectory(executor)
add_subdirectory(graph)
add_subdirectory(model)
add_subdirectory(sampler)
add_subdirectory(speculative)
add_subdirectory(tokenizer)

