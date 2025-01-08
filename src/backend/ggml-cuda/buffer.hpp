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
        m_buffer_type = buffer_type::GGML_GPU;
    }
    
    virtual ~Buffer_CUDA() override {
        if (m_is_cuda_malloc) {
            cuda_context_warp::free_cuda_buffer(m_data_cuda);
        }

        if (m_is_host_malloc) {
            free(m_data_host);
        }
    }
};

}