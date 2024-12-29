add_executable(perplexity_test "main.cpp")
target_link_libraries(perplexity_test PRIVATE smart_serving ggml CLI11::CLI11)

smart_add_artifact(perplexity_test)
