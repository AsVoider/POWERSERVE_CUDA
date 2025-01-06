target_sources(smart_serving PRIVATE
    "platform.cpp"
)

add_subdirectory(ggml)

if (SMART_WITH_QNN)
    add_subdirectory(qnn)
endif()

if (SMART_WITH_CUDA)
    unset(GGML_EXTRA_LIBS_PRIVATE)
    message("here cuda\n")
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "CUDA found")

        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            if (GGML_CUDA_F16 OR GGML_CUDA_DMMV_F16)
                set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75")
            else()
                set(CMAKE_CUDA_ARCHITECTURES "52;61;70;75")
                #set(CMAKE_CUDA_ARCHITECTURES "OFF") # use this to compile much faster, but only F16 models work
            endif()
        endif()
        message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

        enable_language(CUDA)

        file(GLOB   GGML_HEADERS_CUDA "ggml-cuda/*.cuh")
        # list(APPEND GGML_HEADERS_CUDA "../include/ggml-cuda.h")

        file(GLOB   GGML_SOURCES_CUDA "ggml-cuda/*.cu")
        list(APPEND GGML_SOURCES_CUDA "ggml-cuda/ggml-cuda.cpp")
        # list(APPEND GGML_SOURCES_CUDA "ggml-cuda.cu")

        list(APPEND GGML_CDEF_PUBLIC GGML_USE_CUDA)

        add_compile_definitions(GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X})
        add_compile_definitions(GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y})
        add_compile_definitions(K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})
        add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE})

        if (GGML_CUDA_GRAPHS)
            add_compile_definitions(GGML_CUDA_USE_GRAPHS)
        endif()

        if (GGML_CUDA_FORCE_DMMV)
            add_compile_definitions(GGML_CUDA_FORCE_DMMV)
        endif()

        if (GGML_CUDA_FORCE_MMQ)
            add_compile_definitions(GGML_CUDA_FORCE_MMQ)
        endif()

        if (GGML_CUDA_FORCE_CUBLAS)
            add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
        endif()

        if (GGML_CUDA_NO_VMM)
            add_compile_definitions(GGML_CUDA_NO_VMM)
        endif()

        if (DEFINED GGML_CUDA_DMMV_Y)
            add_compile_definitions(GGML_CUDA_MMV_Y=${GGML_CUDA_DMMV_Y}) # for backwards compatibility
        endif()

        if (GGML_CUDA_F16 OR GGML_CUDA_DMMV_F16)
            add_compile_definitions(GGML_CUDA_F16)
        endif()

        if (GGML_CUDA_NO_PEER_COPY)
            add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
        endif()

        # if (GGML_MUSA)
        #     set_source_files_properties(${GGML_SOURCES_CUDA} PROPERTIES LANGUAGE CXX)
        #     foreach(SOURCE ${GGML_SOURCES_CUDA})
        #         set_property(SOURCE ${SOURCE} PROPERTY COMPILE_FLAGS "-x musa -mtgpu --cuda-gpu-arch=mp_22")
        #     endforeach()
        # endif()

        if (GGML_STATIC)
            if (WIN32)
                # As of 12.3.1 CUDA Toolkit for Windows does not offer a static cublas library
                list(APPEND GGML_EXTRA_LIBS_PRIVATE CUDA::cudart_static CUDA::cublas CUDA::cublasLt)
            else ()
                if (GGML_MUSA)
                    list(APPEND GGML_EXTRA_LIBS_PRIVATE MUSA::musart_static MUSA::mublas_static)
                else()
                    list(APPEND GGML_EXTRA_LIBS_PRIVATE CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
                endif()
            endif()
        else()
            if (GGML_MUSA)
                list(APPEND GGML_EXTRA_LIBS_PRIVATE MUSA::musart MUSA::mublas)
            else()
                list(APPEND GGML_EXTRA_LIBS_PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
            endif()
        endif()

        if (GGML_CUDA_NO_VMM)
            # No VMM requested, no need to link directly with the cuda driver lib (libcuda.so)
        else()
            if (GGML_MUSA)
                list(APPEND GGML_EXTRA_LIBS_PRIVATE MUSA::musa_driver) # required by muDeviceGetAttribute(), muMemGetAllocationGranularity(...), ...
            else()
                list(APPEND GGML_EXTRA_LIBS_PRIVATE CUDA::cuda_driver) # required by cuDeviceGetAttribute(), cuMemGetAllocationGranularity(...), ...
            endif()
        endif()
    else()
        message(WARNING "CUDA not found")
    endif()
    # add_subdirectory(ggml-cuda)
endif()

if (SMART_WITH_CUDA)
    set(CUDA_FLAGS -use_fast_math)

    if (GGML_FATAL_WARNINGS)
        list(APPEND CUDA_FLAGS -Werror all-warnings)
    endif()

    if (GGML_ALL_WARNINGS AND NOT MSVC)
        set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
        if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
            list(APPEND NVCC_CMD -ccbin ${CMAKE_CUDA_HOST_COMPILER})
        endif()

        execute_process(
            COMMAND ${NVCC_CMD} -Xcompiler --version
            OUTPUT_VARIABLE CUDA_CCFULLVER
            ERROR_QUIET
        )

        if (NOT CUDA_CCFULLVER MATCHES clang)
            set(CUDA_CCID "GNU")
            execute_process(
                COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
                OUTPUT_VARIABLE CUDA_CCVER
                ERROR_QUIET
            )
        else()
            if (CUDA_CCFULLVER MATCHES Apple)
                set(CUDA_CCID "AppleClang")
            else()
                set(CUDA_CCID "Clang")
            endif()
            string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
        endif()

        message("-- CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

        get_flags(${CUDA_CCID} ${CUDA_CCVER})
        list(APPEND CUDA_CXX_FLAGS ${CXX_FLAGS} ${GF_CXX_FLAGS})  # This is passed to -Xcompiler later
    endif()

    if (NOT MSVC)
        list(APPEND CUDA_CXX_FLAGS -Wno-pedantic)
    endif()
endif()

if (SMART_WITH_CUDA)
    list(APPEND CUDA_CXX_FLAGS ${ARCH_FLAGS})
    list(JOIN   CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)  # pass host compiler flags as a single argument

    if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
        list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
    endif()

    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
endif()

if (SMART_WITH_CUDA) 
    target_include_directories(smart_serving PRIVATE ${PROJECT_SOURCE_DIR}/libs/ggml/src/ggml-cuda/)
    target_sources(smart_serving PRIVATE ${GGML_SOURCES_CUDA} ${GGML_HEADERS_CUDA})
    target_link_libraries(smart_serving PRIVATE ${GGML_EXTRA_LIBS_PRIVATE})
endif()