add_executable(perplexity-test "main.cpp")
target_link_libraries(perplexity-test PRIVATE powerserve ggml CLI11::CLI11)

powerserve_add_artifact(perplexity-test)
