add_executable(run run.cpp)
target_link_libraries(run PRIVATE powerserve CLI11::CLI11)

powerserve_add_binary(run)
