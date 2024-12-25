add_executable(quant_mul_mat "quant_mul_mat.cpp")
target_link_libraries(quant_mul_mat PRIVATE smart_serving ggml)

# add_executable(ggml_multithread "ggml_multithread.cpp")
# target_link_libraries(ggml_multithread PRIVATE smart_serving ggml)

# add_executable(tensor_view_test "tensor_view_test.cpp")
# target_link_libraries(tensor_view_test PRIVATE smart_serving)

if (SMART_WITH_QNN)
    add_executable(qnn_test qnn_test.cpp)
    target_link_libraries(qnn_test PRIVATE smart_serving CLI11::CLI11)
endif()
