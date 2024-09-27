add_executable(demo demo.cpp gguf.hpp gguf.cpp)
target_link_libraries(demo PRIVATE smart_serving)

add_executable(demo_gguf demo_gguf.cpp)
target_link_libraries(demo_gguf PRIVATE smart_serving)