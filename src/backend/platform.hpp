// 1 platform contains N backends (CPU, NPU, GPU...)
#pragma once

#include "backend/ggml/ggml.hpp"

namespace smart {

struct Platform {
public:
    ggml::GGMLBackend ggml_backend;

public:
    Platform(std::shared_ptr<Config> config, int n_threads) : ggml_backend(config, n_threads) {}

    ~Platform() = default;
};

} // namespace smart
