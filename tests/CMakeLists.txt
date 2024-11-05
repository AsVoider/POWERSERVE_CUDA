add_executable(quant_mul_mat "quant_mul_mat.cpp")
target_link_libraries(quant_mul_mat PRIVATE smart_serving ggml)

add_executable(ggml_multithread "ggml_multithread.cpp")
target_link_libraries(ggml_multithread PRIVATE smart_serving ggml)

add_executable(sampler_test "sampler_test.cpp")
target_link_libraries(sampler_test PRIVATE smart_serving)
