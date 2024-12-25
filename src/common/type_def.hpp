#pragma once

#include "llama-vocab.h"

#include <array>
#include <filesystem>

namespace smart {

using Path  = std::filesystem::path;
using Token = llama_vocab::id;

static constexpr size_t max_n_dims = 4;
using Shape                        = std::array<size_t, max_n_dims>;
using Stride                       = std::array<size_t, max_n_dims>;

} // namespace smart
