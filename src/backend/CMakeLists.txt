target_sources(smart_serving PRIVATE
    "platform.cpp"
)

add_subdirectory(ggml)

if (SMART_WITH_QNN)
    add_subdirectory(qnn)
endif()
