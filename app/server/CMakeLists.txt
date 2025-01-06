add_library(server.simple STATIC simple_server.cpp)
target_link_libraries(server.simple PUBLIC
    powerserve
    httplib::httplib
    nlohmann_json::nlohmann_json
    concurrentqueue
)

add_library(server.local STATIC local_server.cpp)
target_link_libraries(server.local PUBLIC
    powerserve
    httplib::httplib
    nlohmann_json::nlohmann_json
    concurrentqueue
)

add_executable(server server.cpp)
target_link_libraries(server PRIVATE
    powerserve
    CLI11::CLI11
    server.simple
)

powerserve_add_binary(server)
