// 1 platform contains N backends (CPU, NPU, GPU...)
#pragma once

#include "backend/ggml/ggml.hpp"
#include "model/llama/llama_model.hpp"
#include <vector>

namespace smart {

struct Platform {
	ggml::GGMLBackend ggml_backend;

	Platform(std::shared_ptr<LlamaConfig> config_) : ggml_backend(config_) {}
	~Platform() = default;
};

} // namespace smart
