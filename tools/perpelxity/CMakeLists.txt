add_executable(perpelxity_test "main.cpp")
target_link_libraries(perpelxity_test PRIVATE smart_serving ggml CLI11::CLI11)
