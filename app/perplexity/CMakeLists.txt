add_executable(perplexity-test "main.cpp")
target_link_libraries(perplexity-test PRIVATE smart_serving ggml CLI11::CLI11)

smart_add_artifact(perplexity-test)
