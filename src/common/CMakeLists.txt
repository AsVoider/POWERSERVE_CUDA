project(smart_serving.common)

file(GLOB_RECURSE common_source_files *.cpp)

add_library(${PROJECT_NAME} STATIC ${common_source_files})

target_link_libraries(${PROJECT_NAME} fmt::fmt)
