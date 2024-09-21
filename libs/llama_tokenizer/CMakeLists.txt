add_library(llama_tokenizer STATIC
    "llama-vocab.cpp"
    "unicode-data.cpp"
    "unicode.cpp"
)
target_link_libraries(llama_tokenizer PRIVATE ggml)
target_include_directories(llama_tokenizer PUBLIC .)
