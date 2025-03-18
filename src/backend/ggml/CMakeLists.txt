target_sources(powerserve PRIVATE
    "ggml.cpp"
    "ggml_kv_cache.cpp"
    "ggml_wrapper.cpp"
    "interface.cpp"
)

target_include_directories(powerserve PRIVATE ${PROJECT_SOURCE_DIR}/libs/ggml/src)