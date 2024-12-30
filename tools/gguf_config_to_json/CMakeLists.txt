add_executable(config-generator "main.cpp")
target_link_libraries(config-generator PRIVATE smart_serving ggml CLI11::CLI11)

smart_add_artifact(config-generator)
