add_executable(demo demo.cpp gguf.hpp gguf.cpp)
target_link_libraries(demo PRIVATE smart_serving)