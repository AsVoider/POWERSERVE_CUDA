#pragma once

#include "core/tensor.hpp"

#include <cstddef>

namespace smart::ggml {

struct Buffer : BaseBuffer {
	using Stride = std::array<size_t, Tensor::max_n_dims>;

	Stride stride; // In bytes
	void *data;
	bool allocated_by_malloc = false;

	Buffer(Stride stride, void *data, bool allocated_by_malloc = false) :
		stride(stride),
		data(data),
		allocated_by_malloc(allocated_by_malloc) {}

	virtual ~Buffer() {
		if (allocated_by_malloc) {
			free(data);
		}
	}
};

} // namespace smart::ggml
