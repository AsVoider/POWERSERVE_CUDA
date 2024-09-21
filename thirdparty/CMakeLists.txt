add_subdirectory(cli11)
add_subdirectory(fmt)

set(GGML_OPENMP OFF)
add_subdirectory(ggml)

if (${CMAKE_C_COMPILER_ID} MATCHES "Clang")
    target_compile_options(ggml PRIVATE -Wno-double-promotion)
endif()

add_subdirectory(llama_tokenizer)
