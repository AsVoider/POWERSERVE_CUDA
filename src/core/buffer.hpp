#pragma once

#include <memory>

namespace smart {

enum class usage : int {
    UNKNOWN = -1,
    ANY = 0,
    WEIGHT = 1,
    COMPUTE = 2,
};

enum class buffer_type : int {
    UNKNOWN = -1,
    GGML_CPU = 0,
    GGML_GPU = 1,
    GGML_GPU_SPLIT = 2,
};

struct BaseBuffer {
public:
    size_t m_size{0UL};
    usage  m_useage{usage::UNKNOWN};
    buffer_type m_buffer_type{buffer_type::UNKNOWN};

public:
    virtual ~BaseBuffer() = default;
};

using BufferPtr = std::shared_ptr<BaseBuffer>;

} // namespace smart
