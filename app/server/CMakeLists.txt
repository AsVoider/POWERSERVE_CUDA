add_library(server.simple STATIC simple_server.cpp)
target_link_libraries(server.simple PUBLIC 
    smart_serving 
    httplib::httplib     
    nlohmann_json::nlohmann_json
    concurrentqueue
)

add_library(server.local STATIC local_server.cpp)
target_link_libraries(server.local PUBLIC 
    smart_serving 
    httplib::httplib     
    nlohmann_json::nlohmann_json
    concurrentqueue
)

add_executable(server server.cpp)
target_link_libraries(server PRIVATE 
    smart_serving 
    CLI11::CLI11 
    server.simple
)

smart_add_binary(server)
