add_executable(gguf_converter "main.cpp")
target_link_libraries(gguf_converter PRIVATE smart_serving ggml CLI11::CLI11)
