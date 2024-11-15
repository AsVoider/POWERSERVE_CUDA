add_executable(config_generator "main.cpp")
target_link_libraries(config_generator PRIVATE smart_serving ggml CLI11::CLI11)

# add_custom_command(
#     TARGET config_generator POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy 
#             $<TARGET_FILE:config_generator>
#             ${CMAKE_BINARY_DIR}/config_generator${CMAKE_EXECUTABLE_SUFFIX}
# )
