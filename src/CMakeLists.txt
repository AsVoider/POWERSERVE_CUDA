add_library(smart_serving STATIC
    "llama_tokenizer.cpp"
)
target_link_libraries(smart_serving PUBLIC fmt::fmt ggml llama_tokenizer)

add_executable(main "main.cpp")
target_link_libraries(main PRIVATE smart_serving CLI11::CLI11)
