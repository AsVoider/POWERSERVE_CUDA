#pragma once

#include "common/logger.hpp"
#include "common/type_def.hpp"
#include "core/buffer.hpp"
#include "core/data_type.hpp"

#include <cstddef>
#include <numeric>

namespace smart {

struct Tensor {
public:
    DataType m_dtype = DataType::UNKNOWN;
    Shape m_shape    = {0};
    BufferPtr m_data = nullptr;

public:
    Tensor()                          = default;
    Tensor(const Tensor &)            = default;
    Tensor &operator=(const Tensor &) = default;

    Tensor(DataType dtype, const Shape &shape) : m_dtype(dtype) {
        SMART_ASSERT(shape.size() <= max_n_dims);
        for (size_t i = 0; i < shape.size(); i++) {
            m_shape[i] = std::max(shape[i], size_t(1));
        }
    }

    Tensor(DataType dtype, Shape &&shape) : m_dtype(dtype), m_shape(std::move(shape)) {
        SMART_ASSERT(shape.size() <= max_n_dims);
    }

public:
    size_t n_dims() const {
        for (size_t i = max_n_dims - 1; i > 0; i--) {
            if (m_shape[i] > 1) {
                return i + 1;
            }
        }
        return 1;
    }

    size_t n_elements() const {
        return static_cast<size_t>(
            std::reduce(std::begin(m_shape), std::end(m_shape), uint64_t(1), std::multiplies<uint64_t>())
        );
    }

    template <typename Buffer>
    auto get() const -> Buffer & {
        return dynamic_cast<Buffer &>(*m_data);
    }

    bool is_quantized() const {
        return m_dtype != DataType::FP32;
    }

    bool is_empty() const {
        for (auto dim : m_shape) {
            if (dim == 0) {
                return true;
            }
        }
        return false;
    }

    int64_t nrows() const {
        return m_shape[1] * m_shape[2] * m_shape[3];
    }

    size_t element_size() const {
        return get_type_size(m_dtype);
    }

    size_t row_size(int64_t ne) const {
        SMART_ASSERT(ne % get_block_size(m_dtype) == 0);
        return element_size() * ne / get_block_size(m_dtype);
    }
};

static inline bool tensor_can_mul_mat(const Tensor *t0, const Tensor *t1) {
    return (t0->m_shape[0] == t1->m_shape[0]) && (t1->m_shape[2] % t0->m_shape[2] == 0) && // verify t0 is broadcastable
           (t1->m_shape[3] % t0->m_shape[3] == 0);
}

static bool tensor_can_repeat(const Tensor *t0, const Tensor *t1) {
    return t0->is_empty() ? t1->is_empty()
                          : (t1->m_shape[0] % t0->m_shape[0] == 0) && (t1->m_shape[1] % t0->m_shape[1] == 0) &&
                                (t1->m_shape[2] % t0->m_shape[2] == 0) && (t1->m_shape[3] % t0->m_shape[3] == 0);
}

} // namespace smart
