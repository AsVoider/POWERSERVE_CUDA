add_executable(config_generator "main.cpp")
target_link_libraries(config_generator PRIVATE smart_serving ggml CLI11::CLI11)

add_artifact(config_generator)
