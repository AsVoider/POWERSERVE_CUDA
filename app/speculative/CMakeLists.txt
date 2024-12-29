add_executable(speculative "speculative.cpp")
target_link_libraries(speculative PRIVATE smart_serving CLI11::CLI11)

smart_add_binary(speculative)
