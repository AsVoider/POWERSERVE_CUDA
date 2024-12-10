add_executable(run "run.cpp")
target_link_libraries(run PRIVATE smart_serving CLI11::CLI11)

add_executable(server "server.cpp")
target_link_libraries(server PRIVATE smart_serving CLI11::CLI11 Crow::Crow)

add_executable(speculative "speculative.cpp")
target_link_libraries(speculative PRIVATE smart_serving CLI11::CLI11)