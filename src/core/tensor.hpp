#pragma once

#include "common.hpp"
#include "core/buffer.hpp"
#include "core/data_type.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>

namespace smart {

struct Tensor {
public:
    static constexpr size_t max_n_dims = 4;

    using Shape = std::array<size_t, max_n_dims>;

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
};

} // namespace smart
