#pragma once

#include "common.hpp"
#include "core/buffer.hpp"
#include "core/data_type.hpp"

#include <array>
#include <numeric>

namespace smart {

struct Tensor {
    static constexpr size_t max_n_dims = 4;

    using Shape = std::array<size_t, max_n_dims>;

    DataType dtype;
    Shape shape;
    BaseBuffer *data = nullptr;

    Tensor(const Tensor &) = default;
    Tensor &operator=(const Tensor &) = default;

    Tensor(DataType dtype_, const Shape &shape_) : dtype(dtype_) {
        std::fill(std::begin(shape), std::end(shape), 1);

        SMART_ASSERT(shape_.size() <= max_n_dims);
        std::copy(shape_.begin(), shape_.end(), std::begin(shape));
    }

    size_t n_dims() const {
        for (size_t i = max_n_dims - 1; i > 0; i--) {
            if (shape[i] > 1) {
                return i + 1;
            }
        }
        return 1;
    }

    size_t n_elements() const {
        return std::reduce(std::begin(shape), std::end(shape), 1, std::multiplies());
    }
};

}
