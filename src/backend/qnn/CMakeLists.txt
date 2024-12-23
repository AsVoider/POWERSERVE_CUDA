set(QNN_SDK $ENV{QNN_SDK_ROOT})

if ("${QNN_SDK}" STREQUAL "")
    message(FATAL_ERROR "Environment variable \"QNN_SDK_ROOT\" is not defined.")
endif()

message("QNN_SDK_ROOT=${QNN_SDK}")

add_library(smart_serving.qnn STATIC
    "causal_models.cpp"
    "graph_interface.cpp"
    "qnn_backend.cpp"
    "qnn.cpp"
)
target_include_directories(smart_serving.qnn PUBLIC ${QNN_SDK}/include/QNN)
target_compile_definitions(smart_serving.qnn PUBLIC SMART_WITH_QNN)

# TODO: We should split out smart_serving.core
target_link_libraries(smart_serving.qnn PUBLIC smart_serving qnn_headers)

target_link_libraries(smart_serving PUBLIC smart_serving.qnn)
