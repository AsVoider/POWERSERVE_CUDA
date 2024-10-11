#pragma once

#include "common.hpp"
#include "core/buffer.hpp"
#include "core/data_type.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <numeric>

namespace smart {

struct Tensor {
	static constexpr size_t max_n_dims = 4;

	using Shape = std::array<size_t, max_n_dims>;

	DataType dtype;
	Shape shape;
	BufferPtr data;

	Tensor(const Tensor &)			  = default;
	Tensor &operator=(const Tensor &) = default;

	Tensor(DataType dtype_, const Shape &shape_) : dtype(dtype_) {
		SMART_ASSERT(shape_.size() <= max_n_dims);
		for (size_t i = 0; i < shape_.size(); i++) {
			shape[i] = std::max(shape_[i], size_t(1));
		}
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
		return static_cast<size_t>(std::reduce(std::begin(shape), std::end(shape), uint64_t(1), std::multiplies<uint64_t>()));
	}

	template <typename Buffer>
	auto get() const -> Buffer & {
		return dynamic_cast<Buffer &>(*data);
	}
};

} // namespace smart
