file(GLOB_RECURSE SMART_SERVING_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/backend/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/backend/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/model/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/model/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sampler/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sampler/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/speculative/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/speculative/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/*.cpp"
)
add_library(smart_serving STATIC ${SMART_SERVING_SOURCES})
target_link_libraries(smart_serving PUBLIC fmt::fmt ggml llama_tokenizer)
target_include_directories(smart_serving PUBLIC .)
