// 1 platform contains N backends (CPU, NPU, GPU...)
#pragma once

#include "backend/ggml/ggml.hpp"

namespace smart {

struct Platform {
	ggml::GGMLBackend ggml_backend;

	Platform(std::shared_ptr<LlamaConfig> config) : ggml_backend(config) {}

	~Platform() = default;
};

} // namespace smart
