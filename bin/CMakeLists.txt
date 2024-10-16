add_executable(run "run.cpp")
target_link_libraries(run PRIVATE smart_serving CLI11::CLI11)