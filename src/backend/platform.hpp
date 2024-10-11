// 1 platform contains N backends (CPU, NPU, GPU...)
#pragma once

#include "backend/ggml/ggml.hpp"
#include "model/llama/llama_model.hpp"
#include <vector>

namespace smart {

struct Platform {
	ggml::GGMLBackend ggml_backend_;

	Platform(std::shared_ptr<LlamaConfig> config) : ggml_backend_(config) {}
	~Platform() = default;
};

} // namespace smart
