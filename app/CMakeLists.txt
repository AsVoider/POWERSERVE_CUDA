set(SMART_BINARY_DIR  ${PROJECT_BINARY_DIR}/bin)
message(STATUS "PowerServe binary dir: ${SMART_BINARY_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${SMART_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${SMART_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${SMART_BINARY_DIR})

function(smart_add_binary target_name)
    smart_add_artifact(${target_name})
endfunction()

add_subdirectory(run)
add_subdirectory(perplexity)
add_subdirectory(server)

if (SMART_WITH_QNN)
    add_subdirectory(speculative)
endif()
