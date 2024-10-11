#pragma once

#include "core/tensor.hpp"
#include <cstddef>
#include <cstdint>

namespace smart::ggml {

struct Buffer : BaseBuffer {
	using Stride = std::array<size_t, Tensor::max_n_dims>;

	Stride stride_; // In bytes
	void *data_;
	bool allocated_by_malloc_ = false;

	Buffer(Stride stride, void *data, bool allocated_by_malloc = false)
		: stride_(stride), data_(data), allocated_by_malloc_(allocated_by_malloc) {}

	virtual ~Buffer() {
		if (allocated_by_malloc_) {
			free(data_);
		}
	}
};

} // namespace smart::ggml
