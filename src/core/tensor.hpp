#pragma once

#include "common.hpp"
#include "core/buffer.hpp"
#include "core/data_type.hpp"

#include <array>
#include <numeric>

namespace smart {

struct Tensor {
    static constexpr size_t n_dims = 4;

    using Shape = std::array<size_t, n_dims>;

    DataType dtype;
    Shape shape;
    BaseBuffer *data = nullptr;

    Tensor(const Tensor &) = default;
    Tensor &operator=(const Tensor &) = default;

    Tensor(DataType dtype_, const Shape &shape_) : dtype(dtype_) {
        std::fill(std::begin(shape), std::end(shape), 1);

        SMART_ASSERT(shape_.size() <= n_dims);
        std::copy(shape_.begin(), shape_.end(), std::begin(shape));
    }

    size_t n_elements() const {
        return std::reduce(std::begin(shape), std::end(shape), 1, std::multiplies());
    }
};

}
