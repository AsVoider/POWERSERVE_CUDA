add_executable(quant_mul_mat "quant_mul_mat.cpp")
target_link_libraries(quant_mul_mat PRIVATE smart_serving ggml)

# add_executable(main "main.cpp")
# target_link_libraries(main PRIVATE smart_serving CLI11::CLI11)