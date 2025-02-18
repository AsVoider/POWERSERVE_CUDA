#pragma once

#include "backend/ggml-cuda/interface.cuh"
#include "core/tensor.hpp"

#include <cstddef>

namespace powerserve::ggml_cuda {

class Buffer_CUDA : public BaseBuffer {
public:
    Stride m_stride;
    void *m_data_cuda{nullptr};
    void *m_data_host{nullptr};
    bool m_is_cuda_malloc{false}; // ? cudaMalloc
    bool m_is_host_malloc{false}; // ? malloc

public:
    Buffer_CUDA(Stride stride, void *data_cuda, void *data_host, usage use, size_t size, bool is_cuda_malloc = false, bool is_host_malloc = false) :
        m_stride{stride}, 
        m_data_cuda{data_cuda}, 
        m_data_host{data_host}, 
        m_is_cuda_malloc{is_cuda_malloc},
        m_is_host_malloc{is_host_malloc} { 
        
        m_useage = use;
        m_size = size;
    }
    
    virtual ~Buffer_CUDA() override {
        if (m_is_cuda_malloc) {
            std::cout << "release cuda" << std::endl;
            cuda_context_warp::free_cuda_buffer(m_data_cuda);
        }

        if (m_is_host_malloc) {
            std::cout << "release host" << std::endl;
            free(m_data_host);
        }
    }

    template <typename T>
    static auto create_buffer(Shape shape) -> BufferPtr {
        Stride stride{};
        stride[0] = sizeof(T);
        for (size_t i{1}; i < shape.size(); ++i) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        size_t size = stride.back() * shape.back();
        
        void *cuda_data_ptr{nullptr};
        cuda_context_warp::malloc_cuda_buffer(&cuda_data_ptr, size);
        return std::make_shared<Buffer_CUDA>(stride, cuda_data_ptr, nullptr, usage::COMPUTE, size, true, false);
    }

    template <typename T>
    static auto create_buffer_view(Buffer_CUDA &p, Shape shape) -> BufferPtr {
        Stride stride{};
        stride[0] = sizeof(T);
        for (size_t i{1}; i < shape.size(); ++i) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        POWERSERVE_ASSERT(p.m_data_cuda != nullptr);
        auto b{std::make_shared<Buffer_CUDA>(stride, nullptr, nullptr, usage::COMPUTE, p.m_size, false, false)};
        b->m_data_cuda = p.m_data_cuda;
        b->m_data_host = p.m_data_host;
        return b;
    }
};

}