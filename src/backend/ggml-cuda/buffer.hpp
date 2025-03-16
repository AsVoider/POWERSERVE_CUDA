#pragma once

#include "backend/ggml-cuda/interface.cuh"
#include "core/tensor.hpp"

#include <cstddef>

namespace powerserve::ggml_cuda {

class Buffer_CUDA : public BaseBuffer {
public:
    Buffer_CUDA(
        Stride stride,
        void *data_device,
        void *data_host,
        usage use,
        size_t size,
        bool is_cuda_malloc = false,
        bool is_host_malloc = false
    ) :
        BaseBuffer{stride, data_device, data_host, is_cuda_malloc, is_host_malloc, size, use} {}

    virtual ~Buffer_CUDA() override {
        if (m_is_device_malloc) {
            cuda_context_warp::free_cuda_buffer_async(m_data_device, default_cuda_context.value());
        }

        if (m_is_host_malloc) {
            free(m_data_host);
        }
    }

    virtual auto get_host_data() -> void * override {
        return m_data_host;
    }

    static auto create_buffer(Shape shape, size_t type_size) -> BufferPtr {
        Stride stride{};
        stride[0] = type_size;
        for (size_t i{1}; i < shape.size(); ++i) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        size_t size = stride.back() * shape.back();

        void *cuda_data_ptr{nullptr};
        // printf("size is %ld, size %% 256 is %ld, size %% 4096 is %ld\n", size, size % 256, size % 4096);
        cuda_data_ptr = default_mempool->allocate(size);
        return std::make_shared<Buffer_CUDA>(stride, cuda_data_ptr, nullptr, usage::COMPUTE, size, false, false);
    }

    static auto create_buffer_view(BaseBuffer &p, Shape shape, size_t type_size, size_t offset = 0) -> BufferPtr {
        Stride stride{};
        stride[0] = type_size;
        for (size_t i{1}; i < shape.size(); ++i) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        auto &parent_buffer{static_cast<Buffer_CUDA &>(p)};
        POWERSERVE_ASSERT(parent_buffer.m_data_device != nullptr);
        auto b{std::make_shared<Buffer_CUDA>(stride, nullptr, nullptr, usage::COMPUTE, p.m_size, false, false)};
        b->m_data_device = static_cast<void *>(static_cast<char *>(parent_buffer.m_data_device) + offset);
        b->m_data_host   = parent_buffer.m_data_host;
        return b;
    }
};

} // namespace powerserve::ggml_cuda
