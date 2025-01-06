add_executable(quant_mul_mat "quant_mul_mat.cpp")
target_link_libraries(quant_mul_mat PRIVATE powerserve ggml)

# add_executable(ggml_multithread "ggml_multithread.cpp")
# target_link_libraries(ggml_multithread PRIVATE powerserve ggml)

# add_executable(tensor_view_test "tensor_view_test.cpp")
# target_link_libraries(tensor_view_test PRIVATE powerserve)

if (POWERSERVE_WITH_QNN)
    add_executable(qnn_test qnn_test.cpp)
    target_link_libraries(qnn_test PRIVATE powerserve CLI11::CLI11)
endif()
