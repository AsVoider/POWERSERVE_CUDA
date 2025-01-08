add_executable(speculative "speculative.cpp")
target_link_libraries(speculative PRIVATE powerserve CLI11::CLI11)

powerserve_add_binary(speculative)
