add_executable(embd_test "test.cpp")
target_link_libraries(embd_test PRIVATE smart_serving ggml CLI11::CLI11)
